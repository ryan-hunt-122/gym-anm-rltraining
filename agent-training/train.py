import gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.ppo import MlpPolicy as PPOMLP
from stable_baselines3.sac import MlpPolicy as SACMLP

from gym_anm.envs import SimplEnv, GenToGridEnv, DelayedSellEnv

from .callbacks import ProgressBarManager, EvalCallback
from .hyperparameters import *
from .utils import parse_args

args = parse_args()

if args.agent == 'PPO':
    MODEL, POLICY = PPO, PPOMLP
elif args.agent == 'SAC':
    MODEL, POLICY = SAC, SACMLP
else:
    raise NotImplementedError

eval_env = DelayedSellEnv()
env = DelayedSellEnv()

eval_callback = EvalCallback(eval_env,
                             best_model_save_path=BASE_DIR,
                             each_model_save_path=BASE_DIR,
                             log_path=BASE_DIR,
                             eval_freq=10000,
                             n_eval_episodes=5
                             )
callbacks = [eval_callback]

model = MODEL(POLICY, env, verbose=0)

with ProgressBarManager(TRAIN_STEPS) as c:
    callbacks += [c]
    model.learn(total_timesteps=TRAIN_STEPS, callback=callbacks)
