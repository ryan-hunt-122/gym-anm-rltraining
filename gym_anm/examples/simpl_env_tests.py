import time

import gym
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from gym_anm import MPCAgentPerfect
from gym_anm.envs.simpl_env.simpl_env import SimplEnv
from gym_anm.envs.battery_env.battery_env import BatteryEnv

from stable_baselines3.sac import SAC
from stable_baselines3.ppo import PPO

def run(algo, save_results = True):
    """

    :param algo: 'SAC' or 'PPO'
    :param save_results:
    :return:
    """

    # env = SimplEnv()
    # env = BatteryEnv()
    env = gym.make('ANM6Easy-v0')
    obs = env.reset()
    done, state = False, None
    observations = []
    actions = []
    T = 300
    total_reward = 0

    print(f'Action space: {env.action_space}')

    model = algo.load("/Users/ryanhunt/PycharmProjects/rl-training/gym-anm-exp/gym_anm/agents/PPO_SimplEnv_v0/best_model")
    # model = MPCAgentPerfect(env.simulator, env.action_space, env.gamma, safety_margin=0.96, planning_steps=10)

    for t in range(T):
        action, state = model.predict(obs, state=state, deterministic=True)
        # action = model.act(env)

        # Scale action to original action space
        # lows = np.array([0, 0, -50, -50])
        # highs = np.array([50, 50, 50, 50])
        # action = lows + (0.5 * (action + 1.0) * (highs - lows))

        obs, reward, done, _ = env.step(action)
        print(f't={t}, r_t={reward:.3}')

        o = obs
        a = action
        actions.append(a)
        observations.append(o)
        total_reward += reward
    print(f'Total Reward: {total_reward}')

    if save_results:
        plot(observations, actions, T)


def plot(observations, actions, T):
    observations = np.transpose(observations)
    actions = np.transpose(actions)

    fig = make_subplots(rows=3, cols=3, start_cell="bottom-left")

    fig.add_trace(go.Scatter(x=[*range(0,T)], y=observations[0], name='Slack P'),
                  row=1, col=1, )

    fig.add_trace(go.Scatter(x=[*range(0,T)], y=observations[1], name='Generator P'),
                  row=1, col=2)

    fig.add_trace(go.Scatter(x=[*range(0,T)], y=observations[2], name='Load P'),
                  row=2, col=1)

    fig.add_trace(go.Scatter(x=[*range(0,T)], y=observations[3], name='Battery P'),
                  row=2, col=2)

    fig.add_trace(go.Scatter(x=[*range(0, T)], y=observations[4], name='Slack Q'),
                  row=1, col=3,)

    fig.add_trace(go.Scatter(x=[*range(0, T)], y=observations[5], name='Generator Q'),
                  row=2, col=3)

    fig.add_trace(go.Scatter(x=[*range(0, T)], y=observations[6], name='Load Q'),
                  row=3, col=1)

    fig.add_trace(go.Scatter(x=[*range(0, T)], y=observations[7], name='Battery Q'),
                  row=3, col=2)

    fig.add_trace(go.Scatter(x=[*range(0, T)], y=observations[8], name='Battery Storage'),
                  row=3, col=3)

    fig.show()

    fig2 = make_subplots(rows=2, cols=2, start_cell="bottom-left")

    fig2.add_trace(go.Scatter(x=[*range(0, T)], y=actions[0], name='Generator P'),
                  row=1, col=1, )

    fig2.add_trace(go.Scatter(x=[*range(0, T)], y=actions[1], name='Generator Q'),
                  row=1, col=2)

    fig2.add_trace(go.Scatter(x=[*range(0, T)], y=actions[2], name='Battery P'),
                  row=2, col=1)

    fig2.add_trace(go.Scatter(x=[*range(0, T)], y=actions[3], name='Battery Q'),
                  row=2, col=2)

    fig2.show()

if __name__ == '__main__':
    ALGO = PPO
    run(ALGO, True)
    print("Done!")