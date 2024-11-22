from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import whisper
import torch
import Levenshtein
import librosa
from whisper_functions import transcribe
import random
import os


class AudioObfuscationEnv(gym.Env):
    def __init__(self, dataset, asr_model):
        super(AudioObfuscationEnv, self).__init__()

        self.dataset = dataset  # List of (audio_file, transcription)
        self.asr_model = asr_model  # Pretrained ASR model
        self.current_index = 0  # Track which file is being used

        self._load_audio_file(self.dataset[self.current_index])

        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=self.audio_signal.shape, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=self.audio_signal.shape, dtype=np.float32)

    def __init__(self, dataset, asr_model):
        super(AudioObfuscationEnv, self).__init__()

        self.dataset = dataset  # List of (audio_file, transcription)
        self.asr_model = asr_model  # Pretrained ASR model
        self.current_index = 0  # Track which file is being used

        # Load the first audio file
        self._load_audio_file(self.dataset[self.current_index])

        self.action_space = spaces.Box(
            low=-0.1, high=0.1, shape=self.audio_signal.shape, dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=self.audio_signal.shape, dtype=np.float32)

    def _load_audio_file(self, data):
        self.audio_signal, _ = librosa.load(data["audio_file"], sr=44100)
        self.transcription = data["transcription"]

    def step(self, action):
        # Apply the action (noise) to the audio
        print("Action: ", action)
        print("Audio Signal: ", self.audio_signal)
        obfuscated_audio = self.audio_signal + action

        # save to file for transcription
        librosa.output.write_wav(
            "obfuscated_audio.wav", obfuscated_audio, 44100)

        # Get transcription from ASR model
        predicted_transcription = transcribe(
            model=self.asr_model, input_file="obfuscated_audio.wav", cuda=False)

        print("Predicted Transcription: ", predicted_transcription)
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


def make_env(dataset, asr_model, rank):
    """
    Returns a function that creates an instance of the environment.
    :param dataset: List of (audio_file, transcription) pairs
    :param asr_model: Pretrained ASR model
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
    audio_files = [f for f in files if f.endswith(".wav")]
    transcriptions = [f.replace(".wav", ".txt") for f in audio_files]
    dataset = [
        {"audio_file": f, "transcription": t} for f, t in zip(audio_files, transcriptions)
    ]
    print("Dataset: ", dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_model = whisper.load_model("base").to(device)

    # Number of parallel environments
    num_envs = 4

    # Create the vectorized environment
    envs = SubprocVecEnv([make_env(dataset, asr_model, i)
                         for i in range(num_envs)])
    # Train the PPO agent
    model = PPO("MlpPolicy", envs, verbose=1)
    model.learn(total_timesteps=100, progress_bar=True)

    # Save the model
    model.save("audio_obfuscation_ppo")
