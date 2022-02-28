"""
This file contains the hyperparameter values used for training and
testing RL agents.
"""

import os

BASE_DIR = './SAC2/'
ENV_ID = 'gym_anm:simplenv-v0'
GAMMA = 0.995

POLICY = 'MlpPolicy'
TRAIN_STEPS = 3000000
MAX_TRAINING_EP_LENGTH = 5000

EVAL_FREQ = 10000
N_EVAL_EPISODES = 5
MAX_EVAL_EP_LENGTH = 3000

LOG_DIR = BASE_DIR + ENV_ID + '/'
os.makedirs(LOG_DIR, exist_ok=True)

TB_LOG_NAME = 'run'


if __name__ == '__main__':
    print('Done.')
