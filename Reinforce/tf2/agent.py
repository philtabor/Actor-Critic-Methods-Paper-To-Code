import tensorflow as tf
import tensorflow.keras as keras
from networks import PolicyNetwork
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam
import numpy as np


class Agent:
    def __init__(self, alpha=0.003, gamma=0.99, n_actions=4,
                 fc1_dims=256, fc2_dims=256, chkpt_dir='models/'):

        self.gamma = gamma
        self.lr = alpha
        self.n_actions = n_actions
        self.chkpt_dir = chkpt_dir
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.policy = PolicyNetwork(n_actions=n_actions, fc1_dims=fc1_dims,
                                    fc2_dims=fc2_dims)
        self.policy.compile(optimizer=Adam(learning_rate=self.lr))

    def save_models(self):
        print('... saving models ...')
        self.policy.save(self.chkpt_dir+'reinforce')

    def load_models(self):
        print('... loading models ...')
        self.policy = keras.models.load_model(self.chkpt_dir+'reinforce')

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        probs = self.policy(state)
        action_probs = tfp.distributions.Categorical(probs=probs)
        action = action_probs.sample()

        return action.numpy()[0]

    def store_transition(self, observation, action, reward):
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)

    def learn(self):
        actions = tf.convert_to_tensor(self.action_memory, dtype=tf.float32)
        rewards = np.array(self.reward_memory)

        G = np.zeros_like(rewards)
        for t in range(len(rewards)):
            G_sum = 0
            discount = 1
            for k in range(t, len(rewards)):
                G_sum += rewards[k] * discount
                discount *= self.gamma
            G[t] = G_sum

        with tf.GradientTape() as tape:
            loss = 0
            for idx, (g, state) in enumerate(zip(G, self.state_memory)):
                state = tf.convert_to_tensor([state], dtype=tf.float32)
                probs = self.policy(state)
                action_probs = tfp.distributions.Categorical(probs=probs)
                log_prob = action_probs.log_prob(actions[idx])
                loss += -g * tf.squeeze(log_prob)
        params = self.policy.trainable_variables
        grads = tape.gradient(loss, params)
        self.policy.optimizer.apply_gradients(zip(grads, params))

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
