import numpy as np
import librosa
from sklearn.decomposition import PCA
from environment.audio_env import AudioObfuscationEnv
from audio.audio import write_waw
from audio.whisper_functions import transcribe

class STFTAudioObfuscationEnv(AudioObfuscationEnv):
    """
    A subclass to overide critical steps for STFT part.
    """
    def step(self, action, sr=44_100):
        """
        Given an action, apply it to the current file and return how well it went.
        """
        # Apply the action (noise) to the audio
        mask = np.array(action).reshape(-1, 1)
        mask = mask.astype(float)
        s_obfuscated = mask * self.magnitude

        # CONVERT BACK TO WAV
        obfuscated_audio = librosa.istft(s_obfuscated * self.phase)

        # save to file for transcription
        write_waw("obfuscated_audio.wav", sr, obfuscated_audio)
        # Get transcription from ASR model
        print("Transcription: ", self.transcription)
        predicted_transcription = transcribe(
            model=self.asr_model, input_file="obfuscated_audio.wav", cuda=False)

        # Calculate reward
        with open(self.transcription, "r") as f:
            actual_transcription = f.read().replace("\n", "")

        transcription_similarity = self._calculate_similarity(
            actual_transcription, predicted_transcription, alpha=4)

        audio_distance = self._noise_reward(obfuscated_audio, 1)
        # Lower similarity and smaller noise are better
        reward = -1*(transcription_similarity+0.1)*(audio_distance+1)
        with open(self._metrics_file, "a") as f:
            f.write(
                f"{self.current_index},{reward},{transcription_similarity},{audio_distance}\n")

        # Define episode termination conditions
        # Single-step environment ends immediately
        print(f"{transcription_similarity=}")
        print(f"{reward=}")

        terminated = transcription_similarity < 0.85
        truncated = False
        info = {}

        # Send FFT signal
        next_state = preprocess_input(obfuscated_audio)
        return next_state, reward, terminated, truncated, info

    
    def perform_attack(self, action, magnitude, phase, sr=44_100):
        mask = np.array(action).reshape(-1, 1)
        mask = mask.astype(float)
        s_obfuscated = mask * magnitude

        # CONVERT BACK TO WAV
        obfuscated_audio = librosa.istft(s_obfuscated * phase)
        return obfuscated_audio
    
    def render(self, mode="human"):
        return NotImplementedError
    

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
