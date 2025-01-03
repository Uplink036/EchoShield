import numpy as np
import librosa
import os
import torch
import whisper
from audio.whisper_functions import transcribe
from environment.audio_env import AudioObfuscationEnv
import soundfile as sf
import scipy.signal as signal
import torch
import numpy as np
import keras
from environment.stft_env import STFTAudioObfuscationEnv, preprocess_input as stft_preprocess_input
from environment.mel_env import MelAudioObfuscationEnv, preprocess_input as mel_preprocess_input
from environment.dol_env import DolAudioObfuscationEnv, preprocess_input as dol_preprocess_input
from models.ddpg import DDPG

WAW_FILEPATH = "data/archive/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)/"
TOTAL_EPISODES = 100
AUDIO_LENGTH = 257

STFT_MODEL_PATH = "stft_trained_model"
MEL_MODEL_PATH = "mel_trained_model"
DOL_MODEL_PATH = "dol_trained_model"
FOLDER = "testing_data/"

STFT_MAGNITUDE = 2
MEL_MAGNITUDE = 500
DOL_MAGNITUDE = 2000
NUM_COMPONENTS = 18
DOL_SAMPLES = 50


def stft_attack(data, sr, env, agent):
    s_full, phase = librosa.magphase(
        librosa.stft(data, n_fft=512))
    prev_state = prev_state = stft_preprocess_input(audio, AUDIO_LENGTH-1, NUM_COMPONENTS)
    tf_prev_state = keras.ops.expand_dims(
        keras.ops.convert_to_tensor(prev_state), 0
    )
    action = agent.policy(tf_prev_state)

    state = env.perform_attack(action, s_full, phase, sr)
    return state

def mel_attack(data, sr, env, agent):
    prev_state = mel_preprocess_input(audio, AUDIO_LENGTH-1, NUM_COMPONENTS)
    tf_prev_state = keras.ops.expand_dims(
        keras.ops.convert_to_tensor(prev_state), 0
    )
    action = agent.policy(tf_prev_state)
    action = action.astype(int) + MEL_MAGNITUDE + 1

    state = env.perform_attack(action, data, sr)

    return state

def dol_attack(data, duration, sr, env, agent):
    prev_state = dol_preprocess_input(audio, AUDIO_LENGTH-1, NUM_COMPONENTS)
    tf_prev_state = keras.ops.expand_dims(
        keras.ops.convert_to_tensor(prev_state), 0
    )
    action = agent.policy(tf_prev_state)

    state = env.perform_attack(action, data, duration, sr)

    return state

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


def append_to_csv(file, index, attack, similarity, audio_distance, audio_path):
    success_threshold = 0.85
    success = 1 if similarity <= success_threshold else 0
    with open(file, "a") as f:
        f.write(f"{index},{attack},{similarity},{audio_distance},{success},{audio_path}\n")


def calculate_measurements(attack, filepath, env, sr, asr_model, original_transcription):
    directory = os.path.dirname(filepath)
    if not os.path.exists(directory):
        os.makedirs(directory)

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
        f.write("file_index,attack_type,similarity,audio_distance,success,audio_path\n")
    # Create the vectorized environment
    env = AudioObfuscationEnv(dataset, asr_model, 0)
    stft_env = STFTAudioObfuscationEnv(dataset, asr_model, AUDIO_LENGTH)
    stft_agent = DDPG(AUDIO_LENGTH, AUDIO_LENGTH, STFT_MAGNITUDE)
    stft_agent.load(STFT_MODEL_PATH)

    mel_env = MelAudioObfuscationEnv(dataset, asr_model, AUDIO_LENGTH)
    mel_agent = DDPG(AUDIO_LENGTH, 2, MEL_MAGNITUDE)
    mel_agent.load(MEL_MODEL_PATH)
    
    dol_env = DolAudioObfuscationEnv(dataset, asr_model, AUDIO_LENGTH)
    dol_agent = DDPG(AUDIO_LENGTH, DOL_SAMPLES, DOL_MAGNITUDE)
    dol_agent.load(DOL_MODEL_PATH)
    metrics_file = "metrics_compare.csv"

    for i in range(10000):
        env.reset()
        sr = env.sample_rate
        audio = env.audio_signal
        with open(env.transcription, "r") as f:
            original_transcription = f.read()

        # STFT attack
        attack_stft = stft_attack(audio, sr, stft_env, stft_agent)
        
        # Mel attack
        attack_mel = mel_attack(audio, sr, mel_env, mel_agent)

        # Dolphin attack
        attack_dol = dol_attack(audio, env.duration, sr, dol_env, dol_agent)

        # Random noise attack
        attack_rn = random_noise_attack(0.01, audio)

        # FFT noise attack
        attack_fn = fft_noise_attack(2, audio)
        # Mel noise attack
        n_fft = 1024
        hop_length = n_fft // 4
        n_mels = 128
        attack_mn = mel_noise_attack(0, audio, sr, n_fft, hop_length, n_mels)

        # Static Dolphin attack
        attack_sd = amplify_dolphin_attack(audio, sr)

        order = 100
        attack_fir = fir_filter_attack(audio, order)

        attack_empty = empty_attack(audio)
        # Transcribe the attack and compare with the original transcription
        folder = "test_attacks/"
        attack_stft_path = "attack_stft/" + str(i) + ".wav"
        attack_mel_path = "attack_mel/" + str(i) + ".wav"
        attack_dol_path = "attack_dol/" + str(i) + ".wav"
        attack_rn_path = "attack_rn/" + str(i) + ".wav"
        attack_fn_path = "attack_fn/" + str(i) + ".wav"
        attack_mn_path = "attack_mn/" + str(i) + ".wav"
        attack_sd_path = "attack_sd/" + str(i) + ".wav"
        attack_fir_path = "attack_fir/" + str(i) + ".wav"
        attack_empty_path = "attack_empty/" + str(i) + ".wav"

        similarity_stft, audio_distance_stft = calculate_measurements(
            attack_stft, folder+attack_stft_path, env, sr, asr_model, original_transcription)
        similarity_mel, audio_distance_mel = calculate_measurements(
            attack_mel, folder+attack_mel_path, env, sr, asr_model, original_transcription)
        similarity_dol, audio_distance_dol = calculate_measurements(
            attack_dol, folder+attack_dol_path, env, sr, asr_model, original_transcription)
        similarity_rn, audio_distance_rn,  = calculate_measurements(
            attack_rn, folder+attack_rn_path, env, sr, asr_model, original_transcription)
        similarity_fn, audio_distance_fn,  = calculate_measurements(
            attack_fn, folder+attack_fn_path, env, sr, asr_model, original_transcription)
        similarity_mn, audio_distance_mn,  = calculate_measurements(
            attack_mn, folder+attack_mn_path, env, sr, asr_model, original_transcription)
        similarity_bp, audio_distance_bp = calculate_measurements(
            attack_sd, folder+attack_sd_path, env, sr, asr_model, original_transcription)
        similarity_fir, audio_distance_fir = calculate_measurements(
            attack_fir, folder+attack_fir_path, env, sr, asr_model, original_transcription)
        similarity_empty, audio_distance_empty = calculate_measurements(
            attack_empty, folder+attack_empty_path, env, sr, asr_model, original_transcription)

        append_to_csv(metrics_file, i, "stft", similarity_stft, audio_distance_stft, attack_stft_path
                      )
        append_to_csv(metrics_file, i, "mel", similarity_mel, audio_distance_mel, attack_mel_path
                      )
        append_to_csv(metrics_file, i, "dol", similarity_dol, audio_distance_dol, attack_dol_path
                      )
        append_to_csv(metrics_file, i, "random_noise",
                      similarity_rn, audio_distance_rn, attack_rn_path)
        append_to_csv(metrics_file, i, "fft_noise",
                      similarity_fn, audio_distance_fn, attack_fn_path)
        append_to_csv(metrics_file, i, "mel_noise",
                      similarity_mn, audio_distance_mn, attack_mn_path)
        append_to_csv(metrics_file, i, "low_and_high_pass",
                      similarity_bp, audio_distance_bp, attack_sd_path)
        append_to_csv(metrics_file, i, "fir_filter",
                      similarity_fir, audio_distance_fir, attack_fir_path)
        append_to_csv(metrics_file, i, "empty",
                      similarity_empty, audio_distance_empty, attack_empty_path)
