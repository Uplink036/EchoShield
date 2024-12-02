import random
import numpy as np
from collections import deque
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, BatchNormalization, Activation, LSTM, Flatten
from keras.optimizers import Adam
import os

# https://domino.ai/blog/deep-reinforcement-learning#body__1eeb210e0437
class DQNAgent:
    def __init__(self, state_size, action_size, action_magnitude=2000):
        self.state_size = state_size
        self.action_size = action_size
        self.action_magnitude = action_magnitude
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = self._build_model()
 
    def _build_model(self):
        input_signal = Input(shape=(self.state_size,1), name="InputSignal")

        x = Conv1D(filters=32, kernel_size=5, strides=2, padding="same")(input_signal)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv1D(filters=64, kernel_size=5, strides=2, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Flatten()(x)
        x = Dense(256, activation="relu")(x)

        output_signal = Dense(self.action_size, activation="relu", name="OutputSignal")(x)

        model = Model(inputs=input_signal, outputs=output_signal)
        model.compile(optimizer="adam", loss="mean_squared_error")

        return model
 
    def remember(self, state, action, reward, next_state, done): 
        self.memory.append((state, action,
                            reward, next_state, done))

    def train(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = state.reshape(1, -1, 1) 
            next_state = next_state.reshape(1, -1, 1) 
            target = reward # if done 
            if not done:
                target = (reward +
                          self.gamma *
                          np.amax(self.model.predict(next_state)))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0) 
        if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.rand() <= self.epsilon: 
            low = -self.action_magnitude
            high = self.action_magnitude
            return np.random.randint(low=low, high=high, size=self.action_size, dtype=np.int16)
        state = state.reshape(1, -1, 1) 
        act_values = self.model.predict(state)
        return act_values.astype(np.int16)

    def save(self, name): 
        self.model.save_weights(name)