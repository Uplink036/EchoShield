import tensorflow as tf
import numpy as np
import keras
from keras import layers
from noise import OUNoise
from buffer import Buffer
from model import AudioObfuscationEnv


class DDPG:
    def __init__(self, state_space, action_space, action_magnitude=100):
        self.actor = get_actor(state_space, action_space, action_magnitude)
        self.critic = get_critic(state_space, action_space)

        self.t_actor = get_actor(state_space, action_space, action_magnitude)
        self.t_critic = get_critic(state_space, action_space)
        self.t_actor.set_weights(self.actor.get_weights())
        self.t_critic.set_weights(self.critic.get_weights())

        self.critic_lr = 0.03
        self.action_lr = 0.01
        self.critic_optimizer = keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = keras.optimizers.Adam(self.action_lr)

        self.noise = OUNoise(state_space, theta=0.015 *
                             action_magnitude, sigma=0.02*action_magnitude)
        self.buffer = Buffer(state_space, action_space)

        self.action_magnitude = action_magnitude
        self.std_dev = 0.2
        self.gamma = 0.99
        self.tau = 0.005
        self.buffer

    def policy(self, state):
        sampled_actions = keras.ops.squeeze(self.actor(state))
        noise = self.noise.sample()
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        legal_action = np.clip(
            sampled_actions, -self.action_magnitude, self.action_magnitude)

        return np.squeeze(legal_action)

    @tf.function
    def update(
        self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = self.t_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.t_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic(
                [state_batch, action_batch], training=True)
            critic_loss = keras.ops.mean(keras.ops.square(y - critic_value))

        critic_grad = tape.gradient(
            critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.t_critic.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = self.actor(state_batch, training=True)
            critic_value = self.critic([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -keras.ops.mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables)
        )

    def learn(self):
        # Get sampling range
        record_range = min(self.buffer.buffer_counter,
                           self.buffer.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.buffer.batch_size)

        # Convert to tensors
        state_batch = keras.ops.convert_to_tensor(
            self.buffer.state_buffer[batch_indices])
        action_batch = keras.ops.convert_to_tensor(
            self.buffer.action_buffer[batch_indices])
        reward_batch = keras.ops.convert_to_tensor(
            self.buffer.reward_buffer[batch_indices])
        reward_batch = keras.ops.cast(reward_batch, dtype="float32")
        next_state_batch = keras.ops.convert_to_tensor(
            self.buffer.next_state_buffer[batch_indices]
        )

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


def get_actor(input_size, output_size, action_magnitude):
    # Initialize weights between -3e-3 and 3-e3
    last_init = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(input_size,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(output_size, activation="relu",
                           kernel_initializer=last_init)(out)

    # Our upper bound is 2.0 for Pendulum.
    outputs = outputs * action_magnitude
    model = keras.Model(inputs, outputs)
    return model


def get_critic(input_size, output_size):
    # State as input
    state_input = layers.Input(shape=(input_size,))
    state_out = layers.Dense(16, activation="relu")(state_input)
    state_out = layers.Dense(32, activation="relu")(state_out)

    # Action as input
    action_input = layers.Input(shape=(input_size,))
    action_out = layers.Dense(output_size, activation="relu")(action_input)

    # Both are passed through separate layer before concatenating
    concat = layers.Concatenate()([state_out, action_out])

    out = layers.Dense(256, activation="relu")(concat)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1)(out)

    # Outputs single value for give state-action
    model = keras.Model([state_input, action_input], outputs)

    return model
