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
        Given an action, ...
        """
        raise NotImplementedError

    def reset(self, *, seed=None, options=None):
        """
        Load the next audio file
        """
        self.current_index = (self.current_index + 1) % len(self.dataset)
        self._load_audio_file(self.dataset[self.current_index]["audio_file"],
                              self.dataset[self.current_index]["transcription"])
        return self.magnitude

    def _calculate_similarity(self, original, predicted, alpha=2):
        """
        Given two sentences, caluclates how similiar they are and returns a number between 0-1.  
        """
        if not isinstance(original, str) or not isinstance(predicted, str):
            raise ValueError(
                f"Invalid inputs: original={original}, predicted={predicted}")
        print(f"Original: {original}, Predicted: {predicted}")
        original = original.lower().strip()
        predicted = predicted.lower().strip()
        return Levenshtein.ratio(original, predicted)**alpha

    def render(self, mode="human"):
        raise NotImplementedError
