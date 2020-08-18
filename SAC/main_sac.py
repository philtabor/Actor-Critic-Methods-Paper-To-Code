# the following 3 lines are helpful if you have multiple GPUs and want to train
# agents on multiple GPUs. I do this frequently when testing.
#import os
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import pybullet_envs
import gym
import numpy as np
from sac_torch import Agent
from utils import plot_learning_curve
import numpy as np

if __name__ == '__main__':
    #env_id = 'LunarLanderContinuous-v2'
    #env_id = 'BipedalWalker-v2'
    #env_id = 'AntBulletEnv-v0'
    env_id = 'InvertedPendulumBulletEnv-v0'
    #env_id = 'CartPoleContinuousBulletEnv-v0'
    env = gym.make(env_id)
    agent = Agent(alpha=0.0003, beta=0.0003, reward_scale=2, env_id=env_id, 
                input_dims=env.observation_space.shape, tau=0.005,
                env=env, batch_size=256, layer1_size=256, layer2_size=256,
                n_actions=env.action_space.shape[0])
    n_games = 250
    filename = env_id + '_'+ str(n_games) + 'games_scale' + str(agent.scale) + \
                    '_clamp_on_sigma.png'
    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []
    load_checkpoint = True
    if load_checkpoint:
        agent.load_models()
        env.render(mode='human')
    steps = 0
    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            steps += 1
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            score += reward
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        print('episode ', i, 'score %.1f' % score,
                'trailing 100 games avg %.1f' % avg_score, 
                'steps %d' % steps, env_id, 
                ' scale ', agent.scale)
    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
