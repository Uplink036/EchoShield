from stable_baselines3.ppo import PPO
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import whisper
import torch
import Levenshtein
from whisper_functions import transcribe
import random
import pandas as pd
import os
from audio import get_wav_info, write_waw
import matplotlib.pyplot as plt


class AudioObfuscationEnv(gym.Env):
    def __init__(self, dataset: list, asr_model: whisper.model):
        super(AudioObfuscationEnv, self).__init__()

        self.dataset = dataset  # List of (audio_file, transcription)
        self.asr_model = asr_model  # Pretrained ASR model
        self.current_index = 0  # Track which file is being used
        self._length_of_file = 3*44100
        # Load the first audio file
        self._load_audio_file(self.dataset[self.current_index])
        self.state = self.audio_signal
        # Define the action and observation spaces
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self._length_of_file,), dtype=np.int16)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self._length_of_file,), dtype=np.int16)

        self._metrics_file = "metrics.csv"

        with open(self._metrics_file, "w") as f:
            f.write("index,reward,transcription_sim,audio_sim\n")

    def _load_audio_file(self, data: dict):
        wav_info = get_wav_info(data["audio_file"])
        self.audio_signal = wav_info["data"][0][0:self._length_of_file]
        if wav_info["length"] < self._length_of_file:
            self.audio_signal = np.pad(
                self.audio_signal, (0, self._length_of_file - len(self.audio_signal)))
        self.transcription = data["transcription"]

    def step(self, action: np.ndarray):
        # Apply the action (noise) to the audio
        action_modifier = 500
        action *= action_modifier
        action = action.astype(np.int16)
        print("Action: ", action)
        print("Audio Signal: ", self.audio_signal)

        obfuscated_audio = self.audio_signal + action
        print("Obfuscated Audio: ", obfuscated_audio)

        # save to file for transcription
        write_waw("obfuscated_audio.wav", 44100, obfuscated_audio)
        # Get transcription from ASR model
        predicted_transcription = transcribe(
            model=self.asr_model, input_file="obfuscated_audio.wav", cuda=False)

        # Calculate reward
        with open(self.transcription, "r") as f:
            actual_transcription = f.read().replace("\n", "")

        transcription_similarity = self._calculate_similarity(
            actual_transcription, predicted_transcription)

        audio_similarity = action_modifier/(np.sum(action)**2+1)
        # Lower similarity and smaller noise are better
        reward = 1-transcription_similarity+audio_similarity
        # Save metrics
        with open(self._metrics_file, "a") as f:
            f.write(f"{self.current_index},{reward},{transcription_similarity},{audio_similarity}\n")
        
        # Define episode termination conditions
        # Single-step environment ends immediately
        terminated = True # Single-step environment
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
        print(f"Original: {original}, Predicted: {predicted}")
        original = original.lower().strip()
        predicted = predicted.lower().strip()
        return Levenshtein.ratio(original, predicted)**2

    def render(self, mode="human"):
        pass

    def display_results(self):
        fig, ax = plt.subplots()
        ax.plot(self.rewards)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        fig.show()


if __name__ == "__main__":
    # Load audio and transcription
    # Preprocessed audio waveform
    files = os.listdir(
        "data/archive/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)")
    audio_files = [
        "data/archive/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)/" + f for f in files if f.endswith(".wav")]
    transcriptions = [f.replace(".wav", ".txt") for f in audio_files]
    dataset = [
        {"audio_file": f, "transcription": t} for f, t in zip(audio_files, transcriptions)
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_model = whisper.load_model("base").to(device)

    # Create the vectorized environment
    env = AudioObfuscationEnv(dataset, asr_model)
    # Train the PPO agent
    steps = 1000
    model = PPO("MlpPolicy", env, verbose=1, n_steps=steps,
                learning_rate=3e-6, batch_size=steps)
    model.learn(total_timesteps=steps, progress_bar=True)
    env.display_results()
    # Save the model
    # model.save("audio_obfuscation_ppo")
