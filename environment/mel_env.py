import numpy as np
import librosa
from sklearn.decomposition import PCA
from environment.audio_env import AudioObfuscationEnv
from audio.audio import write_waw
from audio.whisper_functions import transcribe

class MelAudioObfuscationEnv(AudioObfuscationEnv):
    """
    A subclass to overide critical steps for MEL spectogram part.
    """
    def step(self, action, sr=44_100):
        """
        Given an action, ...
        """
        n_fft = action[0]  # FFT window size
        hop_length = 16  # Hop length
        n_mels = max(action[1] // 8, 64) # Number of Mel bands
        if (n_mels > n_fft):
            n_fft = n_mels

        mel_spec = librosa.feature.melspectrogram(
            y=self.audio_signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        
        obfuscated_audio = librosa.feature.inverse.mel_to_audio(
            mel_spec, sr=sr, n_fft=n_fft, hop_length=hop_length, n_iter=32
            )
        write_waw("obfuscated_audio.wav", sr, obfuscated_audio)
        predicted_transcription = transcribe(
            model=self.asr_model, input_file="obfuscated_audio.wav", cuda=False)
        with open(self.transcription, "r") as f:
            actual_transcription = f.read().replace("\n", "")

        transcription_similarity = self._calculate_similarity(
        actual_transcription, predicted_transcription, alpha=4)

        audio_distance = self._noise_reward(obfuscated_audio, 0.5)
        # Lower similarity and smaller noise are better
        reward = -1*(transcription_similarity+0.1)*(audio_distance+1)
        # Save metrics
        with open(self._metrics_file, "a") as f:
            f.write(
                f"{self.current_index},{reward},{transcription_similarity},{audio_distance}\n")

        terminated = transcription_similarity < 0.85
        truncated = False
        info = {}

        next_state = preprocess_input(obfuscated_audio)
        return next_state, reward, terminated, truncated, info
    
    def perform_attack(self, action, audio, sr=44_100):
        n_fft = action[0]
        hop_length = 16
        n_mels = max(action[1] // 8, 64)
        if (n_mels > n_fft):
            n_fft = n_mels
        
        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        obfuscated_audio = librosa.feature.inverse.mel_to_audio(
            mel_spec, sr=sr, n_fft=n_fft, hop_length=hop_length, n_iter=32
        )
        return obfuscated_audio

def preprocess_input(audio_signal, shape=256, num_components=18):
    """
    Given an audio signal, send back the expected model input
    """
    s_full, _ = librosa.magphase(
        librosa.stft(audio_signal, n_fft=shape*2))
    magnitude = np.array(s_full)
    pca = PCA(n_components=num_components)
    audio_pca = pca.fit_transform(magnitude)
    flat_pca = audio_pca.flatten()
    return flat_pca

