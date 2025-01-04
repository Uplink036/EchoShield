import numpy as np
import librosa
from sklearn.decomposition import PCA
from environment.audio_env import AudioObfuscationEnv, preprocess_input


class STFTAudioObfuscationEnv(AudioObfuscationEnv):
    """
    A subclass to overide critical steps for STFT part.
    """
    def step(self, action, sr=44_100):
        """
        Given an action, apply it to the current file and return how well it went.
        """
        # Apply the action (noise) to the audio
        obfuscated_audio = self.perform_attack(action, self.magnitude, self.phase, sr)
        
        # Calculate reward
        reward, terminated, truncated, info = self.teach(obfuscated_audio, sr, self.reward)
        
        # Send FFT signal
        next_state = preprocess_input(obfuscated_audio)
        return next_state, reward, terminated, truncated, info

    
    def reward(self, transcription_similarity, audio_distance):
        """
        Calculate the reward based on the amount of change to the audio
        """
        reward = -1*(transcription_similarity+0.1)*(audio_distance+1)
        return reward
    
    def perform_attack(self, action, magnitude, phase, sr=44_100):
        mask = np.array(action).reshape(-1, 1)
        mask = mask.astype(float)
        s_obfuscated = mask * magnitude

        # CONVERT BACK TO WAV
        obfuscated_audio = librosa.istft(s_obfuscated * phase)

        return obfuscated_audio
    def render(self, mode="human"):
        return NotImplementedError
    

