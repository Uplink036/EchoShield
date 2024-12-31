import numpy as np
import librosa
import os
import torch
import whisper
from audio.whisper_functions import transcribe
from environment.audio_env import AudioObfuscationEnv
import soundfile as sf
import scipy.signal as signal

FOLDER = "testing_data/"

def get_audio_data(folder_path):
    """
    Given a path, find all files in that path that ends with ".waw" and returns them.
    """
    files = os.listdir(folder_path)
    audio_files = [folder_path + f for f in files if f.endswith(".wav")]
    transcriptions = [f.replace(".wav", ".txt") for f in audio_files]
    dataset = [
        {"audio_file": f, "transcription": t} for f, t in zip(audio_files, transcriptions)
    ]
    return dataset

def random_noise_attack(epsilon, data):
    noise = np.random.uniform(-epsilon, epsilon, data.shape)
    return data + noise


def fft_noise_attack(epsilon, data):
    stft = librosa.stft(data)
    noise = np.random.uniform(-epsilon, epsilon, stft.shape)
    stft_noisy = stft * noise
    return librosa.istft(stft_noisy)


def mel_noise_attack(epsilon, data, sr, n_fft, hop_length, n_mels):
    mel_spec = librosa.feature.melspectrogram(
        y=data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    pertubation = np.random.uniform(-epsilon, epsilon, mel_spec_db.shape)
    perturbed_mel_spec_db = mel_spec_db + pertubation

    perturbed_mel_spec = librosa.db_to_power(perturbed_mel_spec_db)
    reconstructed_audio = librosa.feature.inverse.mel_to_audio(
        perturbed_mel_spec, sr=sr, n_fft=n_fft, hop_length=hop_length, n_iter=1
    )
    return reconstructed_audio


def amplify_dolphin_attack(data, sr):
    # Design band-pass filter (20 Hz to 20,000 Hz)
    band_b, band_a = signal.butter(4, [20, 20000], btype='band', fs=sr)
    band_passed = signal.filtfilt(band_b, band_a, data)
    # Design a low-pass filter for frequencies < 20 Hz
    low_cutoff = 20  # Hz
    low_b, low_a = signal.butter(
        4, low_cutoff / (sr / 2), btype='low')  # 4th-order Butterworth

    # Design a high-pass filter for frequencies > 20,000 Hz
    high_cutoff = 20000  # Hz
    high_b, high_a = signal.butter(4, high_cutoff / (sr / 2), btype='high')

    low_filtered = signal.filtfilt(low_b, low_a, data)
    high_filtered = signal.filtfilt(high_b, high_a, data)

    boost_factor_low = 1000
    boost_factor_high = 1000
    low_boosted = low_filtered * boost_factor_low
    high_boosted = high_filtered * boost_factor_high

    output_signal = low_boosted + high_boosted + band_passed
    # output_signal /= np.max(np.abs(output_signal))
    return output_signal


def fir_filter_attack(data, order):
    # Create a FIR filter with random coefficients
    filter_fir = np.random.uniform(-1, 1, order)
    return np.convolve(data, filter_fir)


def empty_attack(data):
    return data


def append_to_csv(file, index, attack, similarity, audio_distance):
    success_threshold = 0.85
    success = 1 if similarity <= success_threshold else 0
    with open(file, "a") as f:
        f.write(f"{index},{attack},{similarity},{audio_distance},{success}\n")


def calculate_measurements(attack, filepath, env, sr, asr_model, original_transcription):
    sf.write(filepath, attack, sr)
    transcription = transcribe(asr_model, filepath, cuda=False)
    similarity = env._calculate_similarity(
        original_transcription, transcription)
    audio_distance = env._noise_reward(attack, 1.0)
    return similarity, audio_distance


if __name__ == "__main__":
    if not os.path.exists(FOLDER):
        exit("Test folder does not exist.")
    dataset = get_audio_data(FOLDER)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_model = whisper.load_model("base").to(device)

    with open("metrics_compare.csv", "w") as f:
        f.write("file_index,attack_type,similarity,audio_distance,success\n")
    # Create the vectorized environment
    env = AudioObfuscationEnv(dataset, asr_model, 0)
    metrics_file = "metrics_compare.csv"

    for i in range(10000):
        env.reset()
        sr = env.sample_rate
        audio = env.audio_signal
        with open(env.transcription, "r") as f:
            original_transcription = f.read()

        # Random noise attack
        attack_rn = random_noise_attack(0.01, audio)

        # FFT noise attack
        attack_fn = fft_noise_attack(2, audio)
        # Mel noise attack
        n_fft = 1024
        hop_length = n_fft // 4
        n_mels = 128
        attack_mn = mel_noise_attack(0, audio, sr, n_fft, hop_length, n_mels)

        low = 20
        high = 20000
        attack_bp = amplify_dolphin_attack(audio, sr)

        order = 100
        attack_fir = fir_filter_attack(audio, order)

        attack_empty = empty_attack(audio)
        # Transcribe the attack and compare with the original transcription
        folder = "temp_wavs/"
        attack_rn_path = "attack_rn.wav"
        attack_fn_path = "attack_fn.wav"
        attack_mn_path = "attack_mn.wav"
        attack_bp_path = "attack_bp.wav"
        attack_fir_path = "attack_fir.wav"
        attack_empty_path = "attack_empty.wav"

        similarity_rn, audio_distance_rn,  = calculate_measurements(
            attack_rn, folder+attack_rn_path, env, sr, asr_model, original_transcription)
        similarity_fn, audio_distance_fn,  = calculate_measurements(
            attack_fn, folder+attack_fn_path, env, sr, asr_model, original_transcription)
        similarity_mn, audio_distance_mn,  = calculate_measurements(
            attack_mn, folder+attack_mn_path, env, sr, asr_model, original_transcription)
        similarity_bp, audio_distance_bp = calculate_measurements(
            attack_bp, folder+attack_bp_path, env, sr, asr_model, original_transcription)
        similarity_fir, audio_distance_fir = calculate_measurements(
            attack_fir, folder+attack_fir_path, env, sr, asr_model, original_transcription)
        similarity_empty, audio_distance_empty = calculate_measurements(
            attack_empty, folder+attack_empty_path, env, sr, asr_model, original_transcription)

        append_to_csv(metrics_file, i, "random_noise",
                      similarity_rn, audio_distance_rn)
        append_to_csv(metrics_file, i, "fft_noise",
                      similarity_fn, audio_distance_fn)
        append_to_csv(metrics_file, i, "mel_noise",
                      similarity_mn, audio_distance_mn)
        append_to_csv(metrics_file, i, "low_and_high_pass",
                      similarity_bp, audio_distance_bp)
        append_to_csv(metrics_file, i, "fir_filter",
                      similarity_fir, audio_distance_fir)
        append_to_csv(metrics_file, i, "empty",
                      similarity_empty, audio_distance_empty)
