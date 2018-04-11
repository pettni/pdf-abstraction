from best.ltl import formula_to_mdp
from hVI_firm import FIRM
from hVI_models import Rn_Belief_Space, SI_Model, Gaussian_Noise
from hVI_types import Env, Gamma
import best.rss18_functions as rf

import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import copy

# Define Motion and Observation Model
Wx = np.eye(2)
Wu = np.eye(2)
r2_bs = Rn_Belief_Space([-5, -5], [5, 5])
motion_model = SI_Model(0.1)
obs_model = Gaussian_Noise(2)

# Define Regions
regs = OrderedDict()
p1 = rf.vertex_to_poly(np.array([[-3, -2], [-3, 2], [-1, 2], [-1, -2]]))
regs['r1'] = (p1, 0.5, 'obs')
p2 = rf.vertex_to_poly(np.array([[3, -2], [3, 2], [5, 2], [5, -2]]))
regs['r2'] = (p2, 1.0, 'sample')
# Define Null regions with bounds of the space for null output with lowest priority
p3 = rf.vertex_to_poly(np.array([[r2_bs.x_low[0], r2_bs.x_low[1]],
                                 [r2_bs.x_up[0], r2_bs.x_low[1]],
                                 [r2_bs.x_low[0], r2_bs.x_up[1]],
                                 [r2_bs.x_up[0], r2_bs.x_up[1]]]))
regs['null'] = (p3, 1.0, 'null')
output_color = {'r1':'red', 'r2':'green', 'null':'black'}

# Construct belief-MDP for env
env = Env(regs)

# b0 = 1.0/8*np.ones([2**env.n_unknown_regs, 1])
# b3 = np.zeros([2**env.n_unknown_regs, 1])
# b3[2]=1
b2=np.matrix([[0],[1]])
b1=np.matrix([[0.2],[0.8]])
b0=np.matrix([[0.5],[0.5]])

# Construct and Visualize FIRM
fig = plt.figure(0)
ax = fig.add_subplot(111, aspect='equal')
firm = FIRM(r2_bs, motion_model, obs_model, Wx, Wu, regs, output_color, ax)
firm.compute_output_prob()
firm.plot(ax)
# firm.abstract()

# Create DFA
formula = '! obs U sample'
dfsa, dfsa_init, dfsa_final, proplist = formula_to_mdp(formula)
print dfsa
print dfsa_init
print dfsa_final
print proplist

# Hybrid Value Iteration
val=[[Gamma([b0, b1, b2]) for i_q in range(dfsa.N)] for i_v in range(len(firm.nodes))]

# Initialize Value Function to 1_(q_goal)
for i_v in range(len(firm.nodes)):
    for i_q in range(dfsa.N):
        for i_b in range(len(val[i_v][i_q].belief_points)):
            if i_q in dfsa_final:
                val[i_v][i_q].alpha_mat[:, i_b] = 1  #np.zeros([n_regs, 1])
            else:
                val[i_v][i_q].alpha_mat[:, i_b] = 0  #np.zeros([n_regs, 1])


def backup(i_b, i_v, i_q, val):
    b = val[i_v][i_q].belief_points[i_b]
    if i_q in dfsa_final:
        return np.ones([len(b),1])
    max_alpha_b_e = np.zeros([2**env.n_unknown_regs, 1])
    for i_e in range(len(firm.edges[i_v])):
        p_outputs = firm.edge_output_prob[i_v][i_e]
        v_e = firm.edges[i_v][i_e]
        O = env.get_O(firm.nodes[v_e].mean)
        sum_z = np.zeros([2**env.n_unknown_regs, 1])
        for z, info in firm.regs.iteritems():
            if p_outputs[z] == 0:
                continue
            sum_o = np.zeros([2**env.n_unknown_regs, 1])
            for i_o in range(len(O)):
                q_z_o = None
                # if we are in obstacle region and we observe an obstacle/sample
                if regs[z][1]==1:
                    if regs[z][2]=='obs' or regs[z][2]=='sample':
                        q_z_o = np.argmax(dfsa.T(proplist[regs[z][2]])[i_q, :])
                elif regs[z][1]>0:
                    if (regs[z][2]=='obs' or regs[z][2]=='sample') and (env.x_e[i_o] & 2**env.reg_index[z]):
                        q_z_o = np.argmax(dfsa.T(proplist[regs[z][2]])[i_q, :])
                if q_z_o == None:
                    q_z_o = i_q
                # gamma_o_e = np.matrix(np.matlib.repmat(O[i_o, :], len(b), 1)) * np.matrix(val[v_e][q_z_o].alpha_mat)
                gamma_o_e = np.diag(O[i_o, :]) * np.matrix(val[v_e][q_z_o].alpha_mat)
                # if (i_v==0 and i_q == 0 and i_e==1 and b[0]==0 and i==2):
                #     import pdb; pdb.set_trace()
                index = np.argmax(gamma_o_e.T * b)
                sum_o = sum_o + gamma_o_e[:, index]
            sum_z = sum_z + p_outputs[z] * sum_o
        if max_alpha_b_e.T * np.matrix(b) < sum_z.T * np.matrix(b):
            max_alpha_b_e = sum_z
    print max_alpha_b_e
    return max_alpha_b_e


def plot_val(val):
    v_names=['left','center','right']
    q_names=['NoObsNoSample', 'Sample', 'Obs']
    for i_v in range(len(val)):
        for i_q in range(len(val[0])):
            fig = plt.figure(1)
            fig.add_subplot((len(val)), (len(val[0])), (i_v*len(val[0])+i_q))
            for i_alpha in range(val[i_v][i_q].alpha_mat.shape[1]):
                plt.plot(val[i_v][i_q].alpha_mat[:2, i_alpha])
                plt.title('Gamma(v='+v_names[i_v]+',q='+q_names[i_q]+')')
                plt.xlabel('belief', horizontalalignment='right', x=1.0)
                plt.ylabel('Value')
                plt.ylim(-0.5, 1.5)
    plt.show()


val_new = copy.deepcopy(val)
for i in range(10):
    for i_v in range(len(firm.nodes)):
        for i_q in range(dfsa.N):
            for i_b in range(len(val[i_v][i_q].belief_points)):
                alpha_new = backup(i_b, i_v, i_q, val)
                val_new[i_v][i_q].alpha_mat[:, i_b] = np.ravel(alpha_new)
    val = copy.deepcopy(val_new)
plot_val(val_new)
