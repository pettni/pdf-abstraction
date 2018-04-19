from best.ltl import formula_to_mdp
from hVI_firm import FIRM
from hVI_models import Rn_Belief_Space, SI_Model, Gaussian_Noise
from hVI_types import Env, Gamma
import best.rss18_functions as rf

import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import copy
import dill

dill.load_session('val3.pkl')
# Define Motion and Observation Model
Wx = np.eye(2)
Wu = np.eye(2)
r2_bs = Rn_Belief_Space([-5, -5], [5, 5])
motion_model = SI_Model(0.1)
obs_model = Gaussian_Noise(2)

''' Execute the policy '''
b = env.b_prod_init
q = 0
v = 0
traj = []
for t in range(50):
    # Get best edge
    i_best_alpha = np.argmax(val_new[v][q].alpha_mat.T * b)
    best_e = val_new[v][q].best_edge[i_best_alpha]
    # Simulate trajectory under edges
    edge_controller = firm.edge_controllers[v][firm.edges[v].index(best_e)]
    traj_e = edge_controller.simulate_trajectory(edge_controller.node_i)
    traj_n = firm.node_controllers[firm.nodes.index(edge_controller.node_j)].simulate_trajectory(traj_e[-1])
    traj = traj + traj_e + traj_n
    # Get q', v', q' and loop
    z = firm.get_outputs(traj_e + traj_n)
    (b_, o, i_o) = env.get_b_o(traj_n[-1].mean, b, x_e_true)
    v_ = best_e
    if regs[z][1]==1:
        if regs[z][2]=='obs' or regs[z][2]=='sample':
             q_ = np.argmax(dfsa.T(proplist[regs[z][2]])[q, :])
    elif regs[z][1]>0:
        if (regs[z][2]=='obs' or regs[z][2]=='sample') and (env.x_e[i_o] & 2**env.reg_index[z]):
             q_ = np.argmax(dfsa.T(proplist[regs[z][2]])[q, :])
    if q_ == None:
        q_ = q
    b = b_
    q = q_
    v = v_
    print "going to vertex" + str(v)
    print q, v
