"""
This is a file that is meant to bring together other files and test our RL model.
"""

import os
import whisper
import torch
import numpy as np
import keras
from environment.mel_env import MelAudioObfuscationEnv, preprocess_input
from models.ddpg import DDPG
from data_splitting import train_test_split

from audio.audio import get_audio_data
from audio.whisper_functions import get_asr
from models.ddpg import update_target

DATA_FOLDER         = "data/archive/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)/"
TRAINING_FILEPATH   = "training_data/"
TESTING_FILEPATH    = "testing_data/"
RESHUFFLE           = False
RUNS_PER_EPISODE    = 10
TOTAL_EPISODES      = 100
AUDIO_LENGTH        = 257
NUM_COMPONENTS      = 18
OUTPUT_OPTIONS      = 2
ACTION_MAGNITUDE    = 500
SAVE_TRAINED_MODEL  = True
LOAD_TRAINED_MODEL  = False
PATH                = "mel_trained_model"

def train(dataset):
    """
    Trains a DDPG Agent with environment AudioObfuscationEnv
    """
    ep_reward_list = []
    avg_reward_list = []

    env = MelAudioObfuscationEnv(dataset, get_asr(), AUDIO_LENGTH)
    agent = DDPG(AUDIO_LENGTH*NUM_COMPONENTS, OUTPUT_OPTIONS, ACTION_MAGNITUDE)

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
            action = action.astype(int) + ACTION_MAGNITUDE + 1
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

if __name__ == "__main__":
    if not os.path.exists(TRAINING_FILEPATH) or RESHUFFLE:
        train_test_split(DATA_FOLDER, TRAINING_FILEPATH, TESTING_FILEPATH, 0.7)
    dataset = get_audio_data(TRAINING_FILEPATH)
    train(dataset)
