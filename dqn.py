import random
import gym
import numpy as np
from collections import deque
import tensorflow as tf
from keras
from keras.models import Sequential  # type: ignore
from keras.layers import Dense  # type: ignore
from keras.optimizers  # type: ignore
import Adam
import os

# https://domino.ai/blog/deep-reinforcement-learning#body__1eeb210e0437
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()
 
    def _build_model(self):
        model = Sequential() 
        model.add(Dense(32, activation="relu",
                        input_dim=self.state_size))
        model.add(Dense(32, activation="relu"))
        model.add(Dense(self.action_size, activation="linear"))
        model.compile(loss="mse",
                     optimizer=Adam(lr=self.learning_rate))
        return model
 
    def remember(self, state, action, reward, next_state, done): 
        self.memory.append((state, action,
                            reward, next_state, done))

    def train(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward # if done 
            if not done:
                target = (reward +
                          self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0) 
        if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.rand() <= self.epsilon: 
            low = -1
            high = 1
            return np.random.randint(low=low, high=high, size=self.action_size, dtype=np.int16)
        act_values = self.model.predict(state)
        return np.ndarray(act_values, np.int16)

    def save(self, name): 
        self.model.save_weights(name)