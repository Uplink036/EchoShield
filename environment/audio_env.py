"""
Containing all logic releated to the model enivornment and how to handle the audio part of that. 
"""

import whisper
import Levenshtein
import librosa
import numpy as np
import gymnasium as gym
import scipy.spatial.distance as dist
from fastdtw import fastdtw
from audio.audio import get_wav_info, write_waw
from audio.whisper_functions import transcribe


class AudioObfuscationEnv(gym.Env):
    """
    AudioObfuscationEnv, inhertining from the gym.Env class to handle reinforcement learning logic.
    """
    def __init__(self, dataset: list, asr_model: whisper.model, length_of_file):
        self.dataset = dataset  # List of (audio_file, transcription)
        self.asr_model = asr_model  # Pretrained ASR model
        self.current_index = 0  # Track which file is being used
        self._length_of_file = length_of_file
        self._load_audio_file(self.dataset[self.current_index]["audio_file"],
                              self.dataset[self.current_index]["transcription"])
        self._metrics_file = "metrics.csv"

        with open(self._metrics_file, "w") as f:
            f.write("index,reward,transcription_sim,audio_sim\n")

    def _load_audio_file(self, filepath: str, transcription):
        """
        Load an audio file into the memory.
        """
        wav_info = get_wav_info(filepath)
        # Go to FFT
        self.audio_signal = wav_info["data"]
        self.sample_rate = wav_info["samplerate"]
        self.transcription = transcription

        s_full, phase = librosa.magphase(
            librosa.stft(self.audio_signal, n_fft=512))
        self.magnitude = np.array(s_full)
        self.phase = phase

    def _noise_reward(self, obfuscated_audio, alpha=1.0):
        """
        Calculate the reward based on the amount of change to the audio
        """
        mfcc1 = librosa.feature.mfcc(y=self.audio_signal, sr=self.sample_rate, n_mfcc=13)
        mfcc2 = librosa.feature.mfcc(y=obfuscated_audio, sr=self.sample_rate, n_mfcc=13)

        min_frames = min(mfcc1.shape[1], mfcc2.shape[1])
        mfcc_clean = mfcc1[:, :min_frames]
        mfcc_noisy = mfcc2[:, :min_frames]
        # Use Dynamic Time Warping (DTW) for similarity
        distances = np.linalg.norm(mfcc_clean - mfcc_noisy, axis=0)

        # Average distance across all frames
        average_distance = np.mean(distances)*alpha
    
        print(f"Similarity multiplied by alpha: {average_distance}")
        return average_distance
        

    def step(self, action: np.ndarray):
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
        write_waw("obfuscated_audio.wav", 44100, obfuscated_audio)
        # Get transcription from ASR model
        print("Transcription: ", self.transcription)
        predicted_transcription = transcribe(
            model=self.asr_model, input_file="obfuscated_audio.wav", cuda=False)

        # Calculate reward
        with open(self.transcription, "r") as f:
            actual_transcription = f.read().replace("\n", "")

        transcription_similarity = self._calculate_similarity(
            actual_transcription, predicted_transcription)

        audio_distance = self._noise_reward(obfuscated_audio, 0.5)
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
        return action, reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        """
        Load the next audio file
        """
        self.current_index = (self.current_index + 1) % len(self.dataset)
        self._load_audio_file(self.dataset[self.current_index]["audio_file"],
                              self.dataset[self.current_index]["transcription"])
        return self.magnitude

    def _calculate_similarity(self, original, predicted):
        """
        Given two sentences, caluclates how similiar they are and returns a number between 0-1.  
        """
        if not isinstance(original, str) or not isinstance(predicted, str):
            raise ValueError(
                f"Invalid inputs: original={original}, predicted={predicted}")
        print(f"Original: {original}, Predicted: {predicted}")
        original = original.lower().strip()
        predicted = predicted.lower().strip()
        return Levenshtein.ratio(original, predicted)**2

    def render(self, mode="human"):
        raise NotImplementedError
