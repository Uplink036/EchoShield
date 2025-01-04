"""
Handle the creation of DDPG and the logic involved with it. 
"""

import copy
import tensorflow as tf
import numpy as np
import keras
from keras import layers
from models.buffer import Buffer

class DDPG:
    """
    A DDPG Agent, using the actor critic method with OONoise.
    """
    def __init__(self, state_space: int, action_space: int , action_magnitude: float =100):
        """
        Initialize the DDPG agent with the following paramters.

        :param int state_space: Input vector size
        :param int action_space: Output vector size
        :param float action_magnitude: output vector * action_magnitude
        """
        self.actor = get_actor(state_space, action_space, action_magnitude)
        self.critic = get_critic(state_space, action_space)

        self.t_actor = get_actor(state_space, action_space, action_magnitude)
        self.t_critic = get_critic(state_space, action_space)
        self.t_actor.set_weights(self.actor.get_weights())
        self.t_critic.set_weights(self.critic.get_weights())

        self.critic_lr = 0.002
        self.action_lr = 0.001
        self.critic_optimizer = keras.optimizers.Adam(self.critic_lr)
        self.actor_optimizer = keras.optimizers.Adam(self.action_lr)

        self.noise = OUNoise(action_space, theta=0.015*action_magnitude, sigma=0.02*action_magnitude)
        self.buffer = Buffer(state_space, action_space)

        self.action_magnitude = action_magnitude
        self.std_dev = 0.2
        self.gamma = 0.99
        self.tau = 0.005

    def policy(self, state):
        """
        Get an act
        """
        sampled_actions = keras.ops.squeeze(self.actor(state))
        noise = self.noise.sample()
        sampled_actions = sampled_actions.numpy() + noise
        legal_action = np.clip(sampled_actions, -self.action_magnitude, self.action_magnitude)

        return np.squeeze(legal_action)

    @tf.function
    def update(
        self,
        state_batch,
        action_batch,
        reward_batch,
        next_state_batch,
    ):
        """
        Update our actor and critic networks.
        """

        with tf.GradientTape() as tape:
            target_actions = self.t_actor(next_state_batch, training=True)
            y = reward_batch + self.gamma * self.t_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic([state_batch, action_batch], training=True)
            critic_loss = keras.ops.mean(keras.ops.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
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
        """
        Learn from our previous encounter
        """
        record_range = min(self.buffer.buffer_counter, self.buffer.buffer_capacity)
        batch_indices = np.random.choice(record_range, self.buffer.batch_size)

        # Convert to tensors
        state_batch = keras.ops.convert_to_tensor(self.buffer.state_buffer[batch_indices])
        action_batch = keras.ops.convert_to_tensor(self.buffer.action_buffer[batch_indices])
        reward_batch = keras.ops.convert_to_tensor(self.buffer.reward_buffer[batch_indices])
        reward_batch = keras.ops.cast(reward_batch, dtype="float32")
        next_state_batch = keras.ops.convert_to_tensor(
            self.buffer.next_state_buffer[batch_indices]
        )

        self.update(state_batch, action_batch, reward_batch, next_state_batch)
    
    def save(self, path):
        """
        Save the model to a path.
        """
        self.actor.save(path + "actor.keras")
        self.critic.save(path + "critic.keras")
    
    def load(self, path):
        """
        Load the model from a path.
        """
        self.actor = keras.models.load_model(path + "actor.keras")
        self.critic = keras.models.load_model(path + "critic.keras")


def get_actor(input_size, output_size, action_magnitude):
    """
    Get an actor model with additional layers.
    """
    last_init = keras.initializers.RandomUniform(minval=-0.003, maxval=0.003)

    inputs = layers.Input(shape=(input_size,))
    x = layers.Dense(512, activation="relu")(inputs)  # First hidden layer
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(1024, activation="relu")(x)        
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dense(256, activation="relu")(x)       
    outputs = layers.Dense(output_size, activation="tanh",
                           kernel_initializer=last_init)(x)

    # Scale outputs to the desired action magnitude
    outputs = outputs * action_magnitude
    model = keras.Model(inputs, outputs)
    return model


def get_critic(input_size, output_size):
    """
    Get a critic model with additional layers.
    """
    # State as input
    state_input = layers.Input(shape=(input_size,))
    state_out = layers.Dense(64, activation="relu")(
        state_input)  # First state-specific layer
    state_out = layers.Dense(128, activation="relu")(
        state_out)   # Second state-specific layer

    # Action as input
    action_input = layers.Input(shape=(output_size,))
    action_out = layers.Dense(output_size, activation="relu")(
        action_input)  # Action-specific layer

    # Concatenate state and action
    concat = layers.Concatenate()([state_out, action_out])

    x = layers.Dense(512, activation="relu")(concat)  # First combined layer
    x = layers.Dense(512, activation="relu")(x)       
    x = layers.Dense(256, activation="relu")(x)       
    # Single output for state-action value
    outputs = layers.Dense(1)(x)

    model = keras.Model([state_input, action_input], outputs)
    return model

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
    
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = 0
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * \
            np.array([np.random.randn() for i in range(len(x))])
        self.state = x + dx
        return self.state
