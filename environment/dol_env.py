import numpy as np
import librosa
from sklearn.decomposition import PCA
from environment.audio_env import AudioObfuscationEnv, preprocess_input
from audio.audio import write_waw
from audio.whisper_functions import transcribe

class DolAudioObfuscationEnv(AudioObfuscationEnv):
    """
    A subclass to overide critical steps for Dolphin Attack part.
    """
    def step(self, action, sr=44_100):
        """
        Given an action, ...
        """
        time_domain_signal, duration = self.audio_signal, self.duration
    
        obfuscated_audio = self.perform_attack(action, time_domain_signal, duration, sr)

        reward, terminated, truncated, info = self.teach(obfuscated_audio, sr, self.reward)

        next_state = preprocess_input(obfuscated_audio)
        return next_state, reward, terminated, truncated, info
    
    def reward(self, transcription_similarity, audio_distance):
        """
        Calculate the reward based on the amount of change to the audio
        """
        reward = -transcription_similarity
        return reward
    
    def perform_attack(self, action, audio, duration, sr=44_100):
        time_domain_signal = audio 
        
        lowest_freq = 20
        highest_freq = 20_000

        frequency_domain_signal = np.fft.fft(time_domain_signal)

        lowest_sample = int(lowest_freq*duration)
        highest_sample = int(sr*duration/2) # Nyquist frequency

        step_low = int(lowest_sample//10) # 10 steps
        step_high = int((highest_sample-highest_freq*duration)//(len(action)-10)+1) 
        for j, i in enumerate(range(0, lowest_sample, step_low)):
            frequency_domain_signal[i] = frequency_domain_signal[i] + action[j]
        for j, i in enumerate(range(int(highest_freq*duration), highest_sample, step_high)):
            frequency_domain_signal[i] = frequency_domain_signal[i] + action[j+9]
        
        obfuscated_audio = np.fft.ifft(frequency_domain_signal).real
        return obfuscated_audio

