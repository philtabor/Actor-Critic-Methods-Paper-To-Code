import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
from buffer import ReplayBuffer
from networks import ActorNetwork, CriticNetwork, ValueNetwork


class Agent:
    def __init__(self, alpha=0.0003, beta=0.0003, input_dims=[8],
                 env=None, gamma=0.99, n_actions=2, max_size=1000000,
                 tau=0.005, layer1_size=256, layer2_size=256,
                 batch_size=256, reward_scale=2, chkpt_dir='models/'):
        self.gamma = gamma
        self.tau = tau
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.fname = chkpt_dir + 'SAC/'
        self.actor = ActorNetwork(n_actions=n_actions,
                                  max_action=env.action_space.high)
        self.critic_1 = CriticNetwork()
        self.critic_2 = CriticNetwork()
        self.value = ValueNetwork()
        self.target_value = ValueNetwork()

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic_1.compile(optimizer=Adam(learning_rate=beta))
        self.critic_2.compile(optimizer=Adam(learning_rate=beta))
        self.value.compile(optimizer=Adam(learning_rate=beta))
        self.target_value.compile(optimizer=Adam(learning_rate=beta))

        self.scale = reward_scale
        self.update_network_parameters(tau=1)

    def save_models(self):
        # for some environments we can try to save the model before
        # actually calling the learn function. This means we have an empty
        # graph, and TF2 will throw an error
        if self.memory.mem_cntr > self.batch_size:
            print('... saving models ...')
            self.actor.save(self.fname+'actor')
            self.critic_1.save(self.fname+'critic_1')
            self.critic_2.save(self.fname+'critic_2')
            self.value.save(self.fname+'value')
            self.target_value.save(self.fname+'target_value')

    def load_models(self):
        print('... loading models ...')
        self.actor = keras.models.load_model(self.fname+'actor')
        self.critic_1 = keras.models.load_model(self.fname+'critic_1')
        self.critic_2 = keras.models.load_model(self.fname+'critic_2')
        self.value = keras.models.load_model(self.fname+'value')
        self.target_value = keras.models.load_model(self.fname+'target_value')

    def sample_normal(self, state):
        mu, sigma = self.actor(state)
        probabilities = tfp.distributions.Normal(mu, sigma)
        actions = probabilities.sample()  # + something else
        action = tf.math.tanh(actions)*self.actor.max_action
        log_probs = probabilities.log_prob(actions)
        log_probs -= tf.math.log(1-tf.math.pow(action, 2)+self.actor.noise)
        log_probs = tf.math.reduce_sum(log_probs, axis=1, keepdims=True)

        return action, log_probs

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation])
        # actions, _ = self.actor.sample_normal(state)  # reparameterize=False)
        actions, _ = self.sample_normal(state)

        return actions[0]

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_value.weights
        for i, weight in enumerate(self.value.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_value.set_weights(weights)

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
            value = tf.squeeze(self.value(states), 1)

            current_policy_actions, log_probs = self.sample_normal(states)
            # self.actor.sample_normal(states)  # reparameterize=False)
            log_probs = tf.squeeze(log_probs, 1)
            q1_new_pi = self.critic_1((states, current_policy_actions))
            q2_new_pi = self.critic_2((states, current_policy_actions))
            critic_value = tf.squeeze(
                                tf.math.minimum(q1_new_pi, q2_new_pi), 1)

            value_target = critic_value - log_probs
            value_loss = 0.5 * keras.losses.MSE(value, value_target)
        params = self.value.trainable_variables
        grads = tape.gradient(value_loss, params)
        self.value.optimizer.apply_gradients(zip(grads, params))

        with tf.GradientTape() as tape:
            # in the original paper, they reparameterize here. We don't
            # so it's just the usual action.
            new_policy_actions, log_probs = self.sample_normal(states)
            # self.actor.sample_normal(states)  # reparameterize=True)
            log_probs = tf.squeeze(log_probs, 1)
            q1_new_policy = self.critic_1((states, new_policy_actions))
            q2_new_policy = self.critic_2((states, new_policy_actions))
            critic_value = tf.squeeze(tf.math.minimum(
                                        q1_new_policy, q2_new_policy), 1)
            actor_loss = log_probs - critic_value
            actor_loss = tf.math.reduce_mean(actor_loss)
        params = self.actor.trainable_variables
        grads = tape.gradient(actor_loss, params)
        self.actor.optimizer.apply_gradients(zip(grads, params))

        with tf.GradientTape(persistent=True) as tape:
            # I didn't know that these context managers shared values?
            value_ = tf.squeeze(self.target_value(states_), 1)
            q_hat = self.scale*rewards + self.gamma*value_*(1-done)
            q1_old_policy = tf.squeeze(self.critic_1((states, actions)), 1)
            q2_old_policy = tf.squeeze(self.critic_2((states, actions)), 1)
            critic_1_loss = 0.5 * keras.losses.MSE(q1_old_policy, q_hat)
            critic_2_loss = 0.5 * keras.losses.MSE(q2_old_policy, q_hat)
        params_1 = self.critic_1.trainable_variables
        params_2 = self.critic_2.trainable_variables
        grads_1 = tape.gradient(critic_1_loss, params_1)
        grads_2 = tape.gradient(critic_2_loss, params_2)

        self.critic_1.optimizer.apply_gradients(zip(grads_1, params_1))
        self.critic_2.optimizer.apply_gradients(zip(grads_2, params_2))

        self.update_network_parameters()
