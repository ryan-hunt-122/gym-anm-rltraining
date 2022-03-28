import time

import gym
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from gym_anm.envs import DelayedSellEnv

from stable_baselines3.ppo import PPO

from gym_anm.envs.delayedsell_env.delayedsell_env import _get_price_time_series

def run(algo, save_results = True):
    """

    :param algo: 'SAC' or 'PPO'
    :param save_results:
    :return:
    """

    env = DelayedSellEnv()
    obs = env.reset()
    done, state = False, None
    observations = []
    actions = []
    T = 300
    total_reward = 0

    prices = _get_price_time_series()[0]

    print(f'Action space: {env.action_space}')

    # model = PPO.load("/Users/ryanhunt/PycharmProjects/rl-training/gym-anm-exp/gym_anm/plots/delayedsell_env/model/best_model", env=env)
    model = PPO.load("/Users/ryanhunt/PycharmProjects/rl-training/PPO_v0_DSE/all_models/model_2800000")

    for t in range(T):
        action, state = model.predict(obs, state=state, deterministic=True)

        # print(action[2])
        # action[2] = 100*action[2] - 50
        # np.clip(action[2], 0, 100)

        obs, reward, done, _ = env.step(action)
        # print(f't={t}, r_t={reward:.3}')

        o = obs
        a = action
        actions.append(a)
        observations.append(o)
        total_reward += reward
    print(f'Total Reward: {total_reward}')

    if save_results:
        plot(observations, actions, np.tile(prices, 3), T)


def plot(observations, actions, prices, T):
    observations = np.transpose(observations)
    actions = np.transpose(actions)

    fig = make_subplots(rows=3, cols=3, start_cell="bottom-left")

    fig.add_trace(go.Scatter(x=[*range(0,T)], y=observations[0], name='Slack P'),
                  row=1, col=1, )

    fig.add_trace(go.Scatter(x=[*range(0,T)], y=observations[1], name='Generator P'),
                  row=2, col=1)

    fig.add_trace(go.Scatter(x=[*range(0,T)], y=observations[2], name='Battery P'),
                  row=3, col=1)

    fig.add_trace(go.Scatter(x=[*range(0, T)], y=observations[6], name='Battery Storage'),
                  row=3, col=2)

    fig.add_trace(go.Scatter(x=[*range(0, T)], y=prices, name='Prices'),
                  row=1, col=3)

    fig.add_trace(go.Scatter(x=[*range(0, T)], y=actions[0], name='Gen Action'),
                  row=3, col=3)

    fig.add_trace(go.Scatter(x=[*range(0, T)], y=actions[2], name='DES Action'),
                  row=2, col=3)

    fig.show()

    # fig2 = make_subplots(rows=2, cols=2, start_cell="bottom-left")
    #
    # fig2.add_trace(go.Scatter(x=[*range(0, T)], y=actions[0], name='Generator P'),
    #               row=1, col=1, )
    #
    # fig2.add_trace(go.Scatter(x=[*range(0, T)], y=actions[1], name='Generator Q'),
    #               row=1, col=2)

    # fig2.add_trace(go.Scatter(x=[*range(0, T)], y=actions[2], name='Battery P'),
    #               row=2, col=1)
    #
    # fig2.add_trace(go.Scatter(x=[*range(0, T)], y=actions[3], name='Battery Q'),
    #               row=2, col=2)

    # fig2.show()

if __name__ == '__main__':
    ALGO = PPO
    run(ALGO, True)
    print("Done!")