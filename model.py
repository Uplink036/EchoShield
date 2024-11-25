from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import whisper
import torch
import Levenshtein
from whisper_functions import transcribe
import random
import os
from audio import get_wav_info, write_waw


class AudioObfuscationEnv(gym.Env):
    def __init__(self, dataset:list, asr_model:whisper.model):
        super(AudioObfuscationEnv, self).__init__()

        self.dataset = dataset  # List of (audio_file, transcription)
        self.asr_model = asr_model  # Pretrained ASR model
        self.current_index = 0  # Track which file is being used
        self._length_of_file = 3*44100
        # Load the first audio file
        self._load_audio_file(self.dataset[self.current_index])

        self.action_space = spaces.Box(
            low=-0.5, high=0.5, shape=self.audio_signal.shape, dtype=np.int16)
        self.observation_space = spaces.Box(
            low=-5.0, high=5.0, shape=self.audio_signal.shape, dtype=np.int16)

    def _load_audio_file(self, data: dict):
        wav_info = get_wav_info(data["audio_file"])
        self.audio_signal = wav_info["data"][0][0:self._length_of_file]
        if wav_info["length"] < self._length_of_file:
            self.audio_signal = np.pad(
                self.audio_signal, (0, self._length_of_file - len(self.audio_signal)))
        self.transcription = data["transcription"]

    def step(self, action: np.ndarray):
        # Apply the action (noise) to the audio
        print("Action: ", action)
        print("Audio Signal: ", self.audio_signal)
        obfuscated_audio = self.audio_signal + action

        # save to file for transcription
        write_waw("obfuscated_audio.wav", 44100, obfuscated_audio)
        # Get transcription from ASR model
        predicted_transcription = transcribe(
            model=self.asr_model, input_file="obfuscated_audio.wav", cuda=False)

        # Calculate reward
        transcription_similarity = self._calculate_similarity(
            self.transcription, predicted_transcription)
        noise_penalty = np.sum(action ** 2)
        # Lower similarity and smaller noise are better
        reward = -transcription_similarity - noise_penalty

        # Define episode termination conditions
        terminated = True  # Single-step environment ends immediately
        truncated = False  # Not using truncation in this case
        info = {}  # Additional debugging info if needed

        return obfuscated_audio, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Load the next audio file
        self.current_index = (self.current_index + 1) % len(self.dataset)
        self._load_audio_file(self.dataset[self.current_index])
        return self.audio_signal, {}

    def _calculate_similarity(self, original, predicted):
        if not isinstance(original, str) or not isinstance(predicted, str):
            raise ValueError(
                f"Invalid inputs: original={original}, predicted={predicted}")
        return Levenshtein.ratio(original, predicted)

    def render(self, mode="human"):
        pass


def make_env(dataset:list, asr_model:whisper.model, rank:int):
    """
    Returns a function that creates an instance of the environment. \n
    :param dataset: List of (audio_file, transcription) pairs \n
    :param asr_model: Pretrained ASR model \n
    :param rank: Index of the environment (used for debugging/logging)
    """
    def _init():
        env = AudioObfuscationEnv(dataset, asr_model)
        # Distribute audio files across environments
        env.current_index = rank % len(dataset)
        return env
    return _init


if __name__ == "__main__":
    # Load audio and transcription
    # Preprocessed audio waveform
    files = os.listdir(
        "data/archive/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)")
    audio_files = ["data/archive/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)/" + f for f in files if f.endswith(".wav")]
    transcriptions = [f.replace(".wav", ".txt") for f in audio_files]
    dataset = [
        {"audio_file": f, "transcription": t} for f, t in zip(audio_files, transcriptions)
    ]
    print("Dataset: ", dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_model = whisper.load_model("base").to(device)

    # Number of parallel environments
    num_envs = 1

    # Create the vectorized environment
    envs = SubprocVecEnv([make_env(dataset, asr_model, i)
                         for i in range(num_envs)])
    # Train the PPO agent
    model = PPO("MlpPolicy", envs, verbose=0, n_steps=24)
    model.learn(total_timesteps=10, progress_bar=True)

    # Save the model
    model.save("audio_obfuscation_ppo")
