import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, input_dims, alpha=0.001, beta=0.002, env=None,
                 gamma=0.99, n_actions=2, max_size=1000000, tau=0.005,
                 fc1=400, fc2=300, batch_size=64, noise=0.1,
                 chkpt_dir='models/ddpg/'):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.noise = noise
        self.max_action = env.action_space.high[0]
        self.min_action = env.action_space.low[0]
        self.chkpt_dir = chkpt_dir

        self.actor = ActorNetwork(n_actions=n_actions,
                                  fc1_dims=fc1, fc2_dims=fc2)
        self.critic = CriticNetwork(n_actions=n_actions,
                                    fc1_dims=fc1, fc2_dims=fc2)
        self.target_actor = ActorNetwork(n_actions=n_actions,
                                         fc1_dims=fc1, fc2_dims=fc2)
        self.target_critic = CriticNetwork(n_actions=n_actions,
                                           fc1_dims=fc1, fc2_dims=fc2)

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        self.update_network_parameters(tau=1)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save(self.chkpt_dir+'actor')
        self.target_actor.save(self.chkpt_dir+'target_actor')
        self.critic.save(self.chkpt_dir+'critic')
        self.target_critic.save(self.chkpt_dir+'target_critic')

    def load_models(self):
        print('... loading models ...')
        self.actor = keras.models.load_model(self.chkpt_dir+'actor')
        self.target_actor = \
            keras.models.load_model(self.chkpt_dir+'target_actor')
        self.critic = keras.models.load_model(self.chkpt_dir+'critic')
        self.target_critic = \
            keras.models.load_model(self.chkpt_dir+'target_critic')

    def choose_action(self, observation, evaluate=False):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        actions = self.actor(state)
        if not evaluate:
            actions += tf.random.normal(shape=[self.n_actions],
                                        mean=0.0, stddev=self.noise)
        # note that if the env has an action > 1, we have to multiply by
        # max action at some point
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        return actions[0]

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = \
            self.memory.sample_buffer(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(states_)
            critic_value_ = tf.squeeze(self.target_critic(
                                (states_, target_actions)), 1)
            critic_value = tf.squeeze(self.critic((states, actions)), 1)
            target = rewards + self.gamma*critic_value_*(1-done)
            critic_loss = keras.losses.MSE(target, critic_value)
        params = self.critic.trainable_variables
        grads = tape.gradient(critic_loss, params)
        self.critic.optimizer.apply_gradients(zip(grads, params))

        with tf.GradientTape() as tape:
            new_policy_actions = self.actor(states)
            actor_loss = -self.critic((states, new_policy_actions))
            actor_loss = tf.math.reduce_mean(actor_loss)
        params = self.actor.trainable_variables
        grads = tape.gradient(actor_loss, params)
        self.actor.optimizer.apply_gradients(zip(grads, params))

        self.update_network_parameters()
