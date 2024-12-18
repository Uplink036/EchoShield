import numpy
import librosa
import os
import torch
import whisper
from whisper_functions import transcribe
from model import AudioObfuscationEnv
import soundfile as sf

def random_noise_attack(epsilon, data):
    noise = numpy.random.uniform(-epsilon, epsilon, data.shape)
    return data + noise

def fft_noise_attack(epsilon, data):
    stft = librosa.stft(data)
    noise = numpy.random.uniform(-epsilon, epsilon, stft.shape)
    stft_noisy = stft * noise
    return librosa.istft(stft_noisy)

def mel_noise_attack(epsilon, data, sr, n_fft, hop_length, n_mels):
    mel_spec = librosa.feature.melspectrogram(
        y=data, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=numpy.max)
    
    pertubation = numpy.random.uniform(-epsilon, epsilon, mel_spec_db.shape)
    perturbed_mel_spec_db = mel_spec_db + pertubation

    perturbed_mel_spec = librosa.db_to_power(perturbed_mel_spec_db)
    reconstructed_audio = librosa.feature.inverse.mel_to_audio(
        perturbed_mel_spec, sr=sr, n_fft=n_fft, hop_length=hop_length, n_iter=1
    )
    return reconstructed_audio

def band_pass_attack(data, sr, low, high):
    # Remove the frequencies outside the band
    stft = librosa.stft(data)
    stft[:low] = 0
    stft[high:] = 0
    return librosa.istft(stft)
def append_to_csv(file,index, attack, similarity, audio_distance):
    with open(file, "a") as f:
        f.write(f"{index},{attack},{similarity},{audio_distance}\n")

def calculate_measurements(attack, filepath, env, sr, asr_model, original_transcription):
    sf.write(filepath, attack, sr)
    transcription = transcribe(asr_model, filepath, cuda=False)
    similarity = env._calculate_similarity(original_transcription, transcription)
    audio_distance = env._noise_reward(attack, 1.0)
    return similarity, audio_distance

if __name__ == "__main__":
    files = os.listdir(
        "data/archive/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)")
    audio_files = [
        "data/archive/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)/" + f for f in files if f.endswith(".wav")]
    transcriptions = [f.replace(".wav", ".txt") for f in audio_files]
    dataset = [
        {"audio_file": f, "transcription": t} for f, t in zip(audio_files, transcriptions)
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_model = whisper.load_model("base").to(device)

    with open("metrics_compare.csv", "w") as f:
        f.write("file_index,attack_type,similarity,audio_distance\n")
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
        attack_bp = band_pass_attack(audio, sr, low, high)
        # Transcribe the attack and compare with the original transcription
        attack_rn_path = "attack_rn.wav"
        attack_fn_path = "attack_fn.wav"
        attack_mn_path = "attack_mn.wav"
        attack_bp_path = "attack_bp.wav"

        similarity_rn, audio_distance_rn = calculate_measurements(attack_rn, attack_rn_path, env, sr, asr_model, original_transcription)
        similarity_fn, audio_distance_fn = calculate_measurements(attack_fn, attack_fn_path, env, sr, asr_model, original_transcription)
        similarity_mn, audio_distance_mn = calculate_measurements(attack_mn, attack_mn_path, env, sr, asr_model, original_transcription)
        similarity_bp, audio_distance_bp = calculate_measurements(attack_bp, attack_bp_path, env, sr, asr_model, original_transcription)

        append_to_csv(metrics_file, i, "random_noise", similarity_rn, audio_distance_rn)
        append_to_csv(metrics_file, i, "fft_noise", similarity_fn, audio_distance_fn)
        append_to_csv(metrics_file, i, "mel_noise", similarity_mn, audio_distance_mn)
        append_to_csv(metrics_file, i, "band_pass", similarity_bp, audio_distance_bp)
        


        