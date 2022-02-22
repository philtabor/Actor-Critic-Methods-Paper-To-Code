import gym
import numpy as np
from agent import Agent
from utils import plot_learning_curve, manage_memory
# if you have more than 1 gpu, use device '0' or '1' to assign to a gpu
# import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


if __name__ == '__main__':
    manage_memory()
    best_score = -np.inf
    env = gym.make('LunarLander-v2')
    agent = Agent(alpha=0.0005, gamma=0.99, n_actions=env.action_space.n)
    load_checkpoint = False
    if load_checkpoint:
        agent.load_models()

    num_episodes = 1000
    score_history = []

    for i in range(num_episodes):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward)
            observation = observation_
            score += reward
        score_history.append(score)

        if not load_checkpoint:
            agent.learn()
        avg_score = np.mean(score_history[-100:])
        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = score

        print('episode {} score {:.1f} avg score {:.1f}'.
              format(i, score, avg_score))

    filename = 'plots/lunar-lander.png'
    x = [i for i in range(num_episodes)]
    plot_learning_curve(x, score_history, filename)
