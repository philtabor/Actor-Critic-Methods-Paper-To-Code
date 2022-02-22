import gym
import numpy as np
from agent import Agent
from utils import plot_learning_curve, manage_memory
from gym import wrappers

if __name__ == '__main__':
    manage_memory()
    # env = gym.make('LunarLander-v2')
    env = gym.make('CartPole-v0')
    agent = Agent(alpha=1e-5, n_actions=env.action_space.n)
    n_games = 1800
    record_video = False
    load_checkpoint = False

    # do a mkdir video if you want to record video of the agent playing
    if record_video:
        env = wrappers.Monitor(env, 'video',
                               video_callable=lambda episode_id: True,
                               force=True)
    filename = 'cartpole_1e-5_1024x512_1800games.png'

    figure_file = 'plots/' + filename

    best_score = env.reward_range[0]
    score_history = []

    if load_checkpoint:
        agent.load_models()

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            score += reward
            if not load_checkpoint:
                agent.learn(observation, reward, observation_, done)
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()
        print('episode {} score {:.1f} avg score {:.1f}'.format(
              i, score, avg_score))

    if not load_checkpoint:
        x = [i+1 for i in range(n_games)]
        plot_learning_curve(x, score_history, figure_file)
