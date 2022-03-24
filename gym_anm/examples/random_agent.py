"""
This script illustrates how to interact with gym_anm environments. In this example, the agent samples random
actions from the action space of the ANM6Easy-v0 task for 1000 timesteps. Every time a terminal state is reached, the
environment gets reset.

For more information, see https://gym-anm.readthedocs.io/en/latest/topics/using_env.html.
"""
import gym
import time

from stable_baselines3 import PPO


def run():
    env = gym.make('gym_anm:ANM6Easy-v0')
    obs = env.reset()
    done, state = False, None

    model = PPO.load("/Users/ryanhunt/PycharmProjects/rl-training/gym-anm-exp/gym_anm/agents/PPO_SimplEnv_v0/best_model")

    for i in range(1000):
        action, state = model.predict(obs, state=state, deterministic=True)
        obs, reward, done, _ = env.step(action)

        env.render()
        time.sleep(0.5)   # otherwise the rendering is too fast for the human eye

        if done:
            o = env.reset()
    env.close()

if __name__ == '__main__':
    run()