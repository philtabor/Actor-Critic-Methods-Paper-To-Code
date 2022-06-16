import os
import numpy as np

class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions, name = "memory", chkpt_dir='tmp/ddpg'):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, self.name)
        self.checkpoint_file_best = os.path.join(self.checkpoint_dir, "best", self.name)    #places the same filenames, but in a sub-folder called "best"

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        states_ = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

    def save_memory(self):
        print('... saving memory ...')
        np.savez(self.checkpoint_file, 
                states=self.state_memory, 
                actions=self.action_memory, 
                rewards=self.reward_memory, 
                states_=self.new_state_memory, 
                dones=self.terminal_memory)

    def save_best_memory(self):
        print('... saving best memory ...')
        np.savez(self.checkpoint_file_best, 
                states=self.state_memory, 
                actions=self.action_memory, 
                rewards=self.reward_memory, 
                states_=self.new_state_memory, 
                dones=self.terminal_memory)

    def load_memory(self):
        print('... loading memory ...')
        saved_data = np.load(self.checkpoint_file)
        self.state_memory = saved_data['states']
        self.action_memory = saved_data['actions']
        self.reward_memory = saved_data['rewards']
        self.new_state_memory = saved_data['states_']
        self.terminal_memory = saved_data['dones']
