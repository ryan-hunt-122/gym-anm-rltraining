# gym-anm-rltraining - Extension of gym-anm framework 

[Accidentally deleted the old README, will go through and add any useful comments. Gym-anm README can be found in the gym_anm folder]

Two environments are currently implemented:
- ANM6Easy : built-in example [6-bus, 7-device]
- SimplEnv : custom simplified example [2-bus, 4-device]

Code for both of these is found in `/gym_anm/envs/...`.
- `network.py` defines the devices, buses and branches in the network
- `simpl_env.py` extends the `ANMEnv` class. A lot of the functions can be shared with ANM6Easy, there should be no need for it to be too different.


## Training agents

To train agents, run `python -m rl_agents.train <ALGO> -s <SEED>`.
- ALGO is taken from `{SAC, PPO}`, SEED is a random seed.

Take care to check `hyperparameters.py` first, as directories and length of training should be specified.
[This is code from the original creator, I believe there is an issue with the directories here, will try to fix this]

Within `/SAC2/` is a SAC algorithm trained on SimplEnv. 
