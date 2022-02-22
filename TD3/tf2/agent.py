import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, alpha, beta, input_dims, tau, env,
                 gamma=0.99, update_actor_interval=2, warmup=1000,
                 n_actions=2, max_size=1000000, layer1_size=400,
                 layer2_size=300, batch_size=100, noise=0.1,
                 chkpt_dir='models/'):
        self.gamma = gamma
        self.tau = tau
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.learn_step_cntr = 0
        self.time_step = 0
        self.warmup = warmup
        self.n_actions = n_actions
        self.fname = chkpt_dir
        self.update_actor_iter = update_actor_interval

        self.actor = ActorNetwork(layer1_size, layer2_size,
                                  n_actions=n_actions)

        self.critic_1 = CriticNetwork(layer1_size, layer2_size)
        self.critic_2 = CriticNetwork(layer1_size, layer2_size)

        self.target_actor = ActorNetwork(layer1_size, layer2_size,
                                         n_actions=n_actions)
        self.target_critic_1 = CriticNetwork(layer1_size, layer2_size)
        self.target_critic_2 = CriticNetwork(layer1_size, layer2_size)

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.critic_2.compile(optimizer=Adam(learning_rate=beta))

        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.target_critic_2.compile(optimizer=Adam(learning_rate=beta))

        self.noise = noise
        self.update_network_parameters(tau=1)

    def save_models(self):
        if self.memory.mem_cntr > self.batch_size:
            print('... saving models ...')
            self.actor.save(self.fname+'actor')
            self.critic_1.save(self.fname+'critic_1')
            self.critic_2.save(self.fname+'critic_2')
            self.target_actor.save(self.fname+'target_actor')
            self.target_critic_1.save(self.fname+'target_critic_1')
            self.target_critic_2.save(self.fname+'target_critic_2')

    def load_models(self):
        print('... loading models ...')
        self.actor = keras.models.load_model(self.fname+'actor')
        self.critic_1 = keras.models.load_model(self.fname+'critic_1')
        self.critic_2 = keras.models.load_model(self.fname+'critic_2')
        self.target_actor = keras.models.load_model(self.fname+'target_actor')
        self.target_critic_1 = \
            keras.models.load_model(self.fname+'target_critic_1')
        self.target_critic_2 = \
            keras.models.load_model(self.fname+'target_critic_2')

    def choose_action(self, observation):
        if self.time_step < self.warmup:
            mu = np.random.normal(scale=self.noise, size=(self.n_actions,))
        else:
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            # returns a batch size of 1, want a scalar array
            mu = self.actor(state)[0]
        mu_prime = mu + np.random.normal(scale=self.noise)
        mu_prime = tf.clip_by_value(mu_prime, self.min_action, self.max_action)
        self.time_step += 1

        return mu_prime

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, new_states, dones = \
            self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_states, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            target_actions = self.target_actor(states_)
            target_actions = target_actions + \
                tf.clip_by_value(np.random.normal(scale=0.2), -0.5, 0.5)

            target_actions = tf.clip_by_value(target_actions, self.min_action,
                                              self.max_action)

            q1_ = self.target_critic_1((states_, target_actions))
            q2_ = self.target_critic_2((states_, target_actions))

            q1 = tf.squeeze(self.critic_1((states, actions)), 1)
            q2 = tf.squeeze(self.critic_2((states, actions)), 1)

            # shape is [batch_size, 1], want to collapse to [batch_size]
            q1_ = tf.squeeze(q1_, 1)
            q2_ = tf.squeeze(q2_, 1)

            critic_value_ = tf.math.minimum(q1_, q2_)
            # in tf2 only integer scalar arrays can be used as indices
            # and eager exection doesn't support assignment, so we can't do
            # q1_[dones] = 0.0
            target = rewards + self.gamma*critic_value_*(1-dones)
            critic_1_loss = keras.losses.MSE(target, q1)
            critic_2_loss = keras.losses.MSE(target, q2)
        params_1 = self.critic_1.trainable_variables
        params_2 = self.critic_2.trainable_variables
        grads_1 = tape.gradient(critic_1_loss, params_1)
        grads_2 = tape.gradient(critic_2_loss, params_2)

        self.critic_1.optimizer.apply_gradients(zip(grads_1, params_1))
        self.critic_2.optimizer.apply_gradients(zip(grads_2, params_2))

        self.learn_step_cntr += 1

        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        with tf.GradientTape() as tape:
            new_actions = self.actor(states)
            critic_1_value = self.critic_1((states, new_actions))
            actor_loss = -tf.math.reduce_mean(critic_1_value)
        params = self.actor.trainable_variables
        grads = tape.gradient(actor_loss, params)
        self.actor.optimizer.apply_gradients(zip(grads, params))

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic_1.weights
        for i, weight in enumerate(self.critic_1.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_critic_1.set_weights(weights)

        weights = []
        targets = self.target_critic_2.weights
        for i, weight in enumerate(self.critic_2.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_critic_2.set_weights(weights)
