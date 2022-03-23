import gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy

from gym_anm.envs import SimplEnv

from callbacks import ProgressBarManager
from callbacks import EvalCallback
from rl_agents.hyperparameters import *

if __name__ == '__main__':
    eval_env = gym.make('ANM6Easy-v0')
    env = gym.make('ANM6Easy-v0')

    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=BASE_DIR,
                                 each_model_save_path=BASE_DIR,
                                 log_path=BASE_DIR,
                                 eval_freq=10000,
                                 n_eval_episodes=5
                                 )
    callbacks = [eval_callback]

    model = PPO(MlpPolicy, env, verbose=0)
    with ProgressBarManager(TRAIN_STEPS) as c:
        callbacks += [c]
        model.learn(total_timesteps=TRAIN_STEPS, callback=callbacks)

    # print("Evaluating")
    # mean_reward, std_reward = evaluate(model, num_episodes=100)

    # print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
