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
    def __init__(self, state_size, action_size, action_magnitude=10):
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
        input_signal = Input(shape=(self.state_size, 1), name="InputSignal")

        x = Conv1D(filters=32, kernel_size=5, strides=2,
                   padding="same")(input_signal)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv1D(filters=64, kernel_size=5, strides=2, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Flatten()(x)
        x = Dense(256, activation="relu")(x)

        output_signal = Dense(
            self.action_size, activation="relu", name="OutputSignal")(x)

        model = Model(inputs=input_signal, outputs=output_signal)
        model.compile(optimizer="adam", loss="mean_squared_error")

        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action,
                            reward, next_state, done))

    def train(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        states = np.array([state for state, _, _, _, _ in minibatch]).reshape(
            batch_size, -1, 1)
        next_states = np.array(
            [next_state for _, _, _, next_state, _ in minibatch]).reshape(batch_size, -1, 1)
        actions = [action for _, action, _, _, _ in minibatch]
        rewards = [reward for _, _, reward, _, _ in minibatch]
        dones = [done for _, _, _, _, done in minibatch]

        next_q_values = self.model.predict(next_states, batch_size=batch_size)

        targets = []
        for i in range(batch_size):
            target = rewards[i]
            if not dones[i]:
                target += self.gamma * np.amax(next_q_values[i])
            targets.append(target)

        q_values = self.model.predict(states, batch_size=batch_size)

        # Update Q-values for the chosen actions
        for i in range(batch_size):
            q_values[i][actions[i]] = targets[i]

        # Fit the model on the batch
        self.model.fit(states, q_values, batch_size=batch_size,
                       epochs=1, verbose=0)

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state):
        low = -self.action_magnitude
        high = self.action_magnitude
        if np.random.rand() <= self.epsilon:
            act_values = np.random.random(
                size=self.action_size)*self.action_magnitude*2 - self.action_magnitude
        else:
            state = state.reshape(1, -1, 1)
            act_values = self.model.predict(state)[0]
        act_values = np.clip(act_values, low, high)
        return act_values.astype(np.float32)

    def save(self, name):
        self.model.save_weights(name)
