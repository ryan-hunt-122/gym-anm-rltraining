import gym
from stable_baselines3 import PPO

from gym_anm.envs import SimplEnv
from gym_anm.envs import GenToGridEnv

from hyperparameters import *

from callbacks import ProgressBarManager
from callbacks import EvalCallback

if __name__ == '__main__':
    eval_env = GenToGridEnv()
    env = GenToGridEnv()

    eval_callback = EvalCallback(eval_env,
                                 best_model_save_path=BASE_DIR,
                                 each_model_save_path=BASE_DIR,
                                 log_path=BASE_DIR,
                                 eval_freq=10000,
                                 n_eval_episodes=1
                                 )
    callbacks = [eval_callback]

    model = PPO.load(MODEL_DIR, env)
    with ProgressBarManager(TRAIN_STEPS) as c:
        callbacks += [c]
        model.learn(total_timesteps=TRAIN_STEPS, callback=callbacks, reset_num_timesteps=False)