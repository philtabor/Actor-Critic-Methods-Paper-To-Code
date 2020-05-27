import gym
import matplotlib.pyplot as plt
import numpy as np
from control_cartpole_q_learning import Agent

class CartPoleStateDigitizer():
    def __init__(self, bounds=(2.4, 4, 0.209, 4), n_bins=10):
        """  
            bounds - bounds for linear space. Single floating point number for
                     each observation element. Space is from -bound to +bound
                     observation -> x, dx/dt, theta, dtheta/dt
        """
        self.position_space = np.linspace(-1*bounds[0], bounds[0], n_bins)
        self.velocity_space = np.linspace(-1*bounds[1], bounds[1], n_bins)
        self.pole_angle_space = np.linspace(-1*bounds[2], bounds[2], n_bins)
        self.pole_velocity_space = np.linspace(-1*bounds[3], bounds[3], n_bins)
        self.states = self.get_state_space()

    def get_state_space(self):
        states = []
        for i in range(len(self.position_space)+1):
            for j in range(len(self.velocity_space)+1):
                for k in range(len(self.pole_angle_space)+1):
                    for l in range(len(self.pole_velocity_space)+1):
                        states.append((i,j,k,l))
        return states

    def digitize(self, observation):
        x, dx_dt, theta, dtheta_dt = observation
        cart_x = int(np.digitize(x, self.position_space))
        cart_dx_dt = int(np.digitize(dx_dt, self.velocity_space))
        pole_theta = int(np.digitize(theta, self.pole_angle_space))
        pole_dtheta_dt = int(np.digitize(dtheta_dt, self.pole_velocity_space))

        return (cart_x, cart_dx_dt, pole_theta, pole_dtheta_dt)

def plot_learning_curve(scores, x):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of scores')
    plt.show()

if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    n_games = 50000
    eps_dec =  2 / n_games
    digitizer = CartPoleStateDigitizer()
    agent = Agent(lr=0.01, gamma=0.99, n_actions=2, eps_start=1.0,
            eps_end=0.01, eps_dec=eps_dec, state_space=digitizer.states)

    scores = []

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        state = digitizer.digitize(observation)
        while not done:
            action = agent.choose_action(state)
            observation_, reward, done, info = env.step(action)
            state_ = digitizer.digitize(observation_)
            agent.learn(state, action, reward, state_)
            state = state_
            score += reward
        if i % 5000 == 0:
            print('episode ', i, 'score %.1f' % score, 
                  'epsilon %.2f' % agent.epsilon)

        agent.decrement_epsilon()
        scores.append(score)

    x = [i + 1 for i in range(n_games)]
    plot_learning_curve(scores, x)
