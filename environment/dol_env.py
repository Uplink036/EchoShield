import numpy as np
import librosa
from sklearn.decomposition import PCA
from environment.audio_env import AudioObfuscationEnv
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

        write_waw("obfuscated_audio.wav", sr, obfuscated_audio)
        predicted_transcription = transcribe(
            model=self.asr_model, input_file="obfuscated_audio.wav", cuda=False)
        with open(self.transcription, "r") as f:
            actual_transcription = f.read().replace("\n", "")

        transcription_similarity = self._calculate_similarity(
        actual_transcription, predicted_transcription, alpha=1)

        audio_distance = self._noise_reward(obfuscated_audio, 0.5)
        # Lower similarity is better
        reward = -transcription_similarity
        # Save metrics
        with open(self._metrics_file, "a") as f:
            f.write(
                f"{self.current_index},{reward},{transcription_similarity},{audio_distance}\n")

        terminated = transcription_similarity < 0.85
        truncated = False
        info = {}

        next_state = preprocess_input(obfuscated_audio)
        return next_state, reward, terminated, truncated, info
    
    def perform_attack(self, action, audio, duration, sr=44_100):
        time_domain_signal = audio
        duration = self.duration
        
        lowest_freq = 20
        highest_freq = 20500

        frequency_domain_signal = np.fft.fft(time_domain_signal)

        lowest_sample = int(lowest_freq*duration)
        highest_sample = int(sr*duration/2) # Nyquist frequency

        step_low = lowest_sample//10 # 10 steps
        step_high = (highest_sample-highest_freq*duration)//(len(action)-10) 
        for j, i in enumerate(range(0, lowest_sample, step_low)):
            frequency_domain_signal[i] = frequency_domain_signal[i] + action[j]
        for j, i in enumerate(range(int(highest_freq*duration), highest_sample, step_high)):
            frequency_domain_signal[i] = frequency_domain_signal[i] + action[j+9]
        
        obfuscated_audio = np.fft.ifft(frequency_domain_signal).real
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

