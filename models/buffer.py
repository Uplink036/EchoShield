"""
All code releated to the buffer class for use with DDPG.
"""
import numpy as np

class Buffer:
    """
    A buffer class designed to handle and remember data for a DDPG model
    """
    def __init__(self, num_states, num_actions, buffer_capacity: int =100000, batch_size=64):
        """
        :param int num_states: The state space for the model
        :param int num_actions: The action space for the model
        :param int buffer_capacity: Number of "experiences" to store at max
        :param int batch_size: Num of tuples to train on.
        """
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    def record(self, obs_tuple):
        """
        Record information into memory, currently takes (s,a,r,s') as input
        """
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1
