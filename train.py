import os
import whisper
import torch
import numpy as np
from dqn import DQNAgent
from model import AudioObfuscationEnv


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


if __name__ == "__main__":
    dataset = getAudioData(
        "data/archive/Raw JL corpus (unchecked and unannotated)/JL(wav+txt)/")
    audio_length = 257
    env = AudioObfuscationEnv(dataset, getASR(), audio_length)
    agent = DQNAgent(audio_length, audio_length, action_magnitude=10)

    batch_size = 32
    n_episodes = 100

    for e in range(n_episodes):
        state = env.reset()
        done = False
        time = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, _, _ = env.step(action)
            reward = reward if not done else reward  # Need to figure this part out
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, n_episodes-1, time, agent.epsilon))
            time += 1
            if time == 10:
                done = True
        break
        if len(agent.memory) > batch_size:
            agent.train(batch_size)
    # if e % 50 == 0:
    #     agent.save("weights_"+ "{:04d}".format(e) + ".hdf5")
