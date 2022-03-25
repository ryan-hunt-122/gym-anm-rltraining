"""
This defines a 2-bus, 4-device environment.

For more information, see https://gym-anm.readthedocs.io/en/latest/topics/design_new_env.html.
"""

from ..anm_env import ANMEnv
import numpy as np

from .network import network

class GenToGridEnv(ANMEnv):

    def __init__(self):
        observation = 'state'          # observation space
        K = 1                    # number of auxiliary variables
        delta_t = 0.25              # time interval between timesteps
        gamma = 0.995                # discount factor
        lamb = 100                 # penalty weighting hyperparameter
        aux_bounds = np.array([[0, 24 / delta_t - 1]])           # bounds on auxiliary variable (optional)
        costs_clipping = (1, 100)       # reward clipping parameters (optional)

        super().__init__(network, observation, K, delta_t, gamma, lamb,
                         aux_bounds, costs_clipping)

        # Consumption and maximum generation 24-hour time series.
        self.P_maxs = _get_gen_time_series()
        self.G_prices = _get_price_time_series()

    def init_state(self):
        n_dev, n_gen, n_des = 2, 1, 0

        state = np.zeros(2 * n_dev + n_des + n_gen + self.K)

        t_0 = self.np_random.randint(0, int(24 / self.delta_t))
        state[-1] = t_0

        # Non-slack generator (P, Q) injections.
        for idx, (dev_id, p_max) in enumerate(zip([1], self.P_maxs)):
            state[2 * n_dev + n_des + idx] = p_max[t_0]
            state[dev_id] = p_max[t_0]
            state[n_dev + dev_id] = \
                self.np_random.uniform(self.simulator.devices[dev_id].q_min,
                                       self.simulator.devices[dev_id].q_max)

        return state

    def next_vars(self, s_t):
        aux = int((s_t[-1] + 1) % (24 / self.delta_t))

        vars = []
        for p_max in self.P_maxs:
            vars.append(p_max[aux])

        for g_price in self.G_prices:
            vars.append(g_price[aux])

        vars.append(aux)

        return np.array(vars)


def _get_gen_time_series():
    """Return the fixed 24-hour time-series for the generator maximum production."""

    # Device 1 (wind farm).
    s1 = 40 * np.ones(25)
    s12 = np.linspace(36.375, 14.625, 7)
    s2 = 11 * np.ones(13)
    s23 = np.linspace(14.725, 36.375, 7)
    s3 = 40 * np.ones(13)
    P1 = np.concatenate((s1, s12, s2, s23, s3, s23[::-1], s2, s12[::-1],
                         s1[:4]))

    # P_maxs = np.vstack((P1, P2))     # For more than 1
    P_maxs = np.expand_dims(P1, axis=0) # For 1
    assert P_maxs.shape == (1, 96)

    return P_maxs

def _get_price_time_series():
    """Return the fixed 24-hour time-series for the grid energy price."""

    s1 = 1 * np.ones(25)
    s12 = np.linspace(1, 1, 7)
    s2 = 1 * np.ones(13)
    s23 = np.linspace(1, 1, 7)
    s3 = 1 * np.ones(13)
    G1 = np.concatenate((s1, s12, s2, s23, s3, s23[::-1], s2, s12[::-1], s1[:4]))

    G_prices = np.expand_dims(G1, axis=0)
    assert G_prices.shape == (1, 96)

    return G_prices