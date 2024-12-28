"""
This is a file that is meant to bring together other files and test our RL model.
"""

import os
import whisper
import torch
import numpy as np
import keras
from environment.mel_env import MelAudioObfuscationEnv
from models.ddpg import DDPG

WAW_FILEPATH = "data/archive/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)/"
RUNS_PER_EPISODE = 10
TOTAL_EPISODES = 100
AUDIO_LENGTH = 257
OUTPUT_OPTIONS = 2
ACTION_MAGNITUDE = 500

def train():
    """
    Trains a DDPG Agent with environment AudioObfuscationEnv
    """
    ep_reward_list = []
    avg_reward_list = []

    env = MelAudioObfuscationEnv(DATASET, get_asr(), AUDIO_LENGTH)
    agent = DDPG(AUDIO_LENGTH, OUTPUT_OPTIONS, ACTION_MAGNITUDE)

    for ep in range(TOTAL_EPISODES):
        prev_state = env.reset()
        prev_state = np.sum(prev_state, axis=1)/prev_state.shape[1]
        episodic_reward = 0
        loop = 0

        while loop < RUNS_PER_EPISODE:
            tf_prev_state = keras.ops.expand_dims(
                keras.ops.convert_to_tensor(prev_state), 0
            )

            action = agent.policy(tf_prev_state)
            action = action.astype(int) + ACTION_MAGNITUDE + 1
            state, reward, done, truncated, _ = env.step(action)

            agent.buffer.record((prev_state, action, reward, state))
            episodic_reward += reward
            agent.learn()

            update_target(agent.t_actor, agent.actor, agent.tau)
            update_target(agent.t_critic, agent.critic, agent.tau)

            loop += 1
            prev_state = state
        
        agent.noise.reset()
        ep_reward_list.append(episodic_reward)

        avg_reward = np.mean(ep_reward_list[-40:])
        print(f"Episode * {ep} * Avg Reward is ==> {avg_reward}")
        avg_reward_list.append(avg_reward)
    
def get_audio_data(folder_path):
    """
    Given a path, find all files in that path that ends with ".waw" and returns them.
    """
    files = os.listdir(folder_path)
    audio_files = [folder_path + f for f in files if f.endswith(".wav")]
    transcriptions = [f.replace(".wav", ".txt") for f in audio_files]
    dataset = [
        {"audio_file": f, "transcription": t} for f, t in zip(audio_files, transcriptions)
    ]
    return dataset

def get_asr():
    """
    Get an (whisper) Automatic Speech Recognition (ASR) model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_model = whisper.load_model("base").to(device)
    return asr_model



# Based on rate `tau`, which is much less than one.
def update_target(target, original, tau):
    """
    Updates the model weights with the target weights

    :param target: The model to change
    :param original: The model that target will change with
    :param tau: a number between 0-1 that will be the change of the model
    """
    target_weights = target.get_weights()
    original_weights = original.get_weights()

    for index, target_weight in enumerate(target_weights):
        target_weights[index] = original_weights[index] * tau + target_weight * (1 - tau)

    target.set_weights(target_weights)

DATASET = get_audio_data(WAW_FILEPATH)
if __name__ == "__main__":
    train()
