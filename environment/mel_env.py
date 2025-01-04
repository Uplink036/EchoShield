import numpy as np
import librosa
from sklearn.decomposition import PCA
from environment.audio_env import AudioObfuscationEnv, preprocess_input

class MelAudioObfuscationEnv(AudioObfuscationEnv):
    """
    A subclass to overide critical steps for MEL spectogram part.
    """
    def step(self, action, sr=44_100):
        """
        Given an action, ...
        """
        obfuscated_audio = self.perform_attack(action, self.audio_signal, sr)
        
        reward, terminated, truncated, info = self.teach(obfuscated_audio, sr, self.reward)

        next_state = preprocess_input(obfuscated_audio)
        return next_state, reward, terminated, truncated, info
    
    def reward(self, transcription_similarity, audio_distance):
        """
        Calculate the reward based on the amount of change to the audio
        """
        reward = -1*(transcription_similarity+0.1)*(audio_distance+1)
        return reward
    
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
