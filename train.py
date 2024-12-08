import os
import whisper
import torch
import numpy as np
import keras
from dqn import DQNAgent, preprocess_input
from model import AudioObfuscationEnv
from ddpg import DDPG


def getAudioData(folderPath):
    files = os.listdir(folderPath)
    audio_files = [folderPath + f for f in files if f.endswith(".wav")]
    transcriptions = [f.replace(".wav", ".txt") for f in audio_files]
    dataset = [
        {"audio_file": f, "transcription": t} for f, t in zip(audio_files, transcriptions)
    ]
    return dataset


def getASR():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr_model = whisper.load_model("base").to(device)
    return asr_model


# Based on rate `tau`, which is much less than one.
def update_target(target, original, tau):
    target_weights = target.get_weights()
    original_weights = original.get_weights()

    for i in range(len(target_weights)):
        target_weights[i] = original_weights[i] * \
            tau + target_weights[i] * (1 - tau)

    target.set_weights(target_weights)


if __name__ == "__main__":
    ep_reward_list = []
    avg_reward_list = []
    total_episodes = 100
    dataset = getAudioData(
        "data/archive/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)/")
    audio_length = 257
    env = AudioObfuscationEnv(dataset, getASR(), audio_length)
    agent = DDPG(audio_length, audio_length, 1)

    for ep in range(total_episodes):
        prev_state = env.reset()
        prev_state = np.sum(prev_state, axis=1)
        episodic_reward = 0

        loop = 0
        while True:
            tf_prev_state = keras.ops.expand_dims(
                keras.ops.convert_to_tensor(prev_state), 0
            )

            action = agent.policy(tf_prev_state)
            # Receive state and reward from environment.
            state, reward, done, truncated, _ = env.step(action)

            agent.buffer.record((prev_state, action, reward, state))
            episodic_reward += reward

            agent.learn()

            update_target(agent.t_actor, agent.actor, agent.tau)
            update_target(agent.t_critic, agent.critic, agent.tau)

            # End this episode when `done` or `truncated` is True

            loop += 1
            if done or truncated or loop == 10:
                agent.noise.reset()
                break

            prev_state = state

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
        avg_reward_list.append(avg_reward)
