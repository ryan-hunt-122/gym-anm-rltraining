"""The input dictionary for a 2-bus, 4-device distribution network."""

import numpy as np

##### CASE FILE DESCRIPTION #####

### Metadata ###
# baseMVA: base power of the system (MVA).

### 1. Bus data:
# BUS_I: bus number (0-indexing).
# BUS_TYPE: bus type (1 = PQ, 2 = PV, 3 = slack).
# BASE_KV : base voltage of the zone containing the bus (kV).
# VMAX: maximum voltage magnitude (p.u.).
# VMIN: minimum voltage magnitude (p.u

### 3. Branch data.
# F_BUS: "from" bus number.
# T_BUS: "to" bus number.
# BR_R: resistance (p.u.).
# BR_X: reactance (p.u.).
# BR_B: total line charging susceptance (p.u.).
# RATE: MVA rating.
# TAP: transformer off-nominal turns ratio. If non-zero, taps located at
# 'from' bus and impedance at 'to' bus (see pi-model); if zero, indicating
# no-transformer (i_from.e. a simple transmission line).
# SHIFT: transformer phase shit angle (degrees), positive => delay.
# BR_STATUS: branch status, 1 = in service, 0 = out-of-service.

network = {"baseMVA": 100.0}

network["bus"] = np.array([
    [0, 0, 132, 1., 1.],
    [1, 1, 33, 1.1, 0.9]
])

network["device"] = np.array([
    [0, 0, 0, None, 200, -200, 200, -200, None, None, None, None, None, None, None],
    [1, 1, 2, None, 50, 0, 50, -50, 35, None, 20, -20, None, None, None],
    [2, 1, -1, 0.2, 0, -30, None, None, None, None, None, None, None, None, None],
    [3, 1, 3, None, 50, -50, 50, -50, 30, -30, 25, -25, 100, 0, 0.9]
])

network["branch"] = np.array([
    [0, 1, 0.03, 0.022, 0., 25, 1, 0]
])