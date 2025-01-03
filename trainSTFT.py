"""
This is a file that is meant to bring together other files and test our RL model.
"""

import os
import whisper
import torch
import numpy as np
import keras
from environment.stft_env import STFTAudioObfuscationEnv, preprocess_input
from models.ddpg import DDPG
from data_splitting import train_test_split

DATA_FOLDER         = "data/archive/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)/"
TRAINING_FILEPATH   = "training_data/"
TESTING_FILEPATH    = "testing_data/"
RESHUFFLE           = False
TOTAL_EPISODES      = 150
AUDIO_LENGTH        = 257
NUM_COMPONENTS      = 18
RUNS_PER_EPISODE    = 20
SAVE_TRAINED_MODEL  = True
LOAD_TRAINED_MODEL  = False
PATH                = "stft_trained_model"

def train(dataset):
    """
    Trains a DDPG Agent with environment AudioObfuscationEnv
    """
    ep_reward_list = []
    avg_reward_list = []

    env = STFTAudioObfuscationEnv(dataset, get_asr(), AUDIO_LENGTH)
    agent = DDPG(AUDIO_LENGTH*NUM_COMPONENTS, AUDIO_LENGTH, 2)
    if LOAD_TRAINED_MODEL:
        agent.load(PATH)

    for ep in range(TOTAL_EPISODES):
        audio = env.reset()
        prev_state = preprocess_input(audio, AUDIO_LENGTH-1, NUM_COMPONENTS)
        episodic_reward = 0
        loop = 0

        while loop < RUNS_PER_EPISODE:
            tf_prev_state = keras.ops.expand_dims(
                keras.ops.convert_to_tensor(prev_state), 0
            )

            action = agent.policy(tf_prev_state)
            state, reward, done, truncated, _ = env.step(action)

            agent.buffer.record((prev_state, action, reward, state))
            episodic_reward += reward
            agent.learn()

            update_target(agent.t_actor, agent.actor, agent.tau)
            update_target(agent.t_critic, agent.critic, agent.tau)

            prev_state = state
            loop += 1

        agent.noise.reset()
        ep_reward_list.append(episodic_reward)

        avg_reward = np.mean(ep_reward_list[-40:])
        print(f"Episode * {ep} * Avg Reward is ==> {avg_reward}")
        avg_reward_list.append(avg_reward)

    if SAVE_TRAINED_MODEL:
        agent.save(PATH)


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
        target_weights[index] = original_weights[index] * \
            tau + target_weight * (1 - tau)

    target.set_weights(target_weights)

if __name__ == "__main__":
    if not os.path.exists(TRAINING_FILEPATH) or RESHUFFLE:
        train_test_split(DATA_FOLDER, TRAINING_FILEPATH, TESTING_FILEPATH, 0.7)
    dataset = get_audio_data(TRAINING_FILEPATH)
    train(dataset)
