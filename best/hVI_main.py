from best.ltl import formula_to_mdp
from hVI_firm import FIRM
from hVI_models import Rn_Belief_Space, SI_Model, Gaussian_Noise
from hVI_types import Env, Gamma
import best.rss18_functions as rf

import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import copy
import cPickle as pkl
import random
from os import fork
from joblib import Parallel, delayed
import multiprocessing

''' Configuration Parameters '''
# sc = 'toy'
sc = 'rss'
load = False  # Reads value function from pickle
parr = True
random.seed(10)
np.random.seed(10)
epsilon = 1e-8

# Define Motion and Observation Model
Wx = np.eye(2)
Wu = np.eye(2)
r2_bs = Rn_Belief_Space([-5, -5], [5, 5])
motion_model = SI_Model(0.1)
obs_model = Gaussian_Noise(2)

''' TODO:
- Smarter expansion of belief set to make algorithm anytime
'''

print "Setting Up Scenario"
# Define Regions
#### Toy
if sc == 'toy':
    regs = OrderedDict()
    p1 = rf.vertex_to_poly(np.array([[-3, -2], [-3, 2], [-1, 2], [-1, -2]]))
    regs['r1'] = (p1, 0.5, 'obs')
    p2 = rf.vertex_to_poly(np.array([[3, -2], [3, 2], [5, 2], [5, -2]]))
    regs['r2'] = (p2, 1.0, 'sample')
    output_color = {'r1':'red', 'r2':'green', 'null':'white'}
    #Define Null regions with bounds of the space for null output with lowest priority
    p = rf.vertex_to_poly(np.array([[r2_bs.x_low[0], r2_bs.x_low[1]],
                                    [r2_bs.x_up[0], r2_bs.x_low[1]],
                                    [r2_bs.x_low[0], r2_bs.x_up[1]],
                                    [r2_bs.x_up[0], r2_bs.x_up[1]]]))
    regs['null'] = (p, 1.0, 'null')
    b_set = []
    b_set.append(np.matrix([[0.5],[0.5]]))
    b_set.append(np.matrix([[0.2],[0.8]]))
    b_set.append(np.matrix([[0.8],[0.2]]))
    b_set.append(np.matrix([[0.95],[0.05]]))
    b_set.append(np.matrix([[0.05],[0.95]]))
    b_set.append(np.matrix([[0],[1]]))
    b_set.append(np.matrix([[1],[0]]))
    env = Env(regs)
    x_e_true = env.get_product_belief([0])

#### RSS
elif sc == 'rss':
    regs = OrderedDict()
    p1 = rf.vertex_to_poly(np.array([[1.2, 0], [2.2, 1], [-1.6, 3.6], [-2.6, 2.6]]))
    regs['r1'] = (p1, 1, 'obs')
    p2 = rf.vertex_to_poly(np.array([[-3, 4], [-3, 5], [-5, 5], [-5, 4]]))
    regs['r2'] = (p2, 1, 'obs')
    p3 = rf.vertex_to_poly(np.array([[2, -1.5], [3, -1], [5, -3], [5, -5], [4, -5]]))
    regs['r3'] = (p3, 1, 'obs')
    p4 = rf.vertex_to_poly(np.array([[1.2, 0], [2.2, 1], [2, -1.5], [3, -1]]))
    regs['r4'] = (p4, 0.5, 'obs')
    p5 = rf.vertex_to_poly(np.array([[2, -1.5], [2.5, -2.5], [1, -5], [-1, -5]]))
    regs['r5'] = (p5, 1, 'obs')

    a1 = rf.vertex_to_poly(np.array([[4, -2], [5, -2], [5, -1], [4, -1]]))
    regs['a1'] = (a1, 0.9, 'sample')
    a2 = rf.vertex_to_poly(np.array([[2, -4], [3, -4], [3, -5], [2, -5]]))
    regs['a2'] = (a2, 0.0, 'sample')
    a3 = rf.vertex_to_poly(np.array([[-2, 0], [-2, 1], [-1, 1], [-1, 0]]))
    regs['a3'] = (a3, 0.1, 'sample')

    output_color = {'r1':'red', 'r2':'red', 'r3':'red', 'r4':'orange', 'r5':'orange',
                'a1':'green', 'a2':'green', 'a3':'green', 'null':'white'}
    # Define Null regions with bounds of the space for null output with lowest priority
    p = rf.vertex_to_poly(np.array([[r2_bs.x_low[0], r2_bs.x_low[1]],
                                    [r2_bs.x_up[0], r2_bs.x_low[1]],
                                    [r2_bs.x_low[0], r2_bs.x_up[1]],
                                    [r2_bs.x_up[0], r2_bs.x_up[1]]]))
    regs['null'] = (p, 1.0, 'null')
    # Construct belief-MDP for env
    env = Env(regs)
    b_set = []
    probs = [0, 0.2, 0.5, 0.8, 1]
    # b_set = [env.get_product_belief([b1, b2, b3, b4, b5]) for b1 in probs for b2 in probs for b3 in probs for b4 in probs for b5 in probs]
    # x_e_true = env.get_product_belief([0, 0, 0, 0, 1])
    b_set = [env.get_product_belief([b1, b2, b3]) for b1 in probs for b2 in probs for b3 in probs]
    x_e_true = env.get_product_belief([0, 1, 0])
else:
    raise ValueError('Invalid Environment')


# Construct and Visualize FIRM
print "Constructing FIRM"
fig = plt.figure(0)
ax = fig.add_subplot(111, aspect='equal')
firm = FIRM(r2_bs, motion_model, obs_model, Wx, Wu, regs, output_color, ax, sc)
firm.compute_output_prob()
firm.plot(ax)
# firm.abstract()
# Create DFA
formula = '! obs U sample'
dfsa, dfsa_init, dfsa_final, proplist = formula_to_mdp(formula)


def backup(i_b, i_v, i_q, val):
    b = val[i_v][i_q].belief_points[i_b]
    if i_q in dfsa_final:
        return (np.ones([len(b), 1]), firm.edges[i_v][0]) # return 0th edge caz it doesn't matter
    index_alpha_init = np.argmax(val[i_v][i_q].alpha_mat.T * b)  # set max alpha to current max
    max_alpha_b_e = val[i_v][i_q].alpha_mat[:, index_alpha_init]
    best_e = val[i_v][i_q].best_edge[i_b]
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
                    # if regs[z][2]=='obs' or regs[z][2]=='sample':
                    if regs[z][2]=='obs' or regs[z][2]=='sample' and regs[z][0].contains(firm.nodes[v_e].mean):
                        q_z_o = np.argmax(dfsa.T(proplist[regs[z][2]])[i_q, :])
                elif regs[z][1]>0:
                    # if (regs[z][2]=='obs' or regs[z][2]=='sample') and (env.x_e[i_o] & 2**env.reg_index[z] == 2**env.reg_index[z]):
                    if (regs[z][2]=='obs' or regs[z][2]=='sample') and (env.x_e[i_o] & 2**env.reg_index[z] == 2**env.reg_index[z]) and regs[z][0].contains(firm.nodes[v_e].mean):
                        q_z_o = np.argmax(dfsa.T(proplist[regs[z][2]])[i_q, :])
                if q_z_o == None:
                    q_z_o = i_q
                gamma_o_e = np.diag(np.ravel(O[i_o, :])) * np.matrix(val[v_e][q_z_o].alpha_mat)
                index = np.argmax(gamma_o_e.T * b)
                sum_o = sum_o + gamma_o_e[:, index]
                # if (i_v==8 and i_q==0 and (v_e==7) and i==3):
                    # print "obs = " + str(i_o) + "q_z_o = " + str(q_z_o)
                    # print sum_z.T * b
                    # print max_alpha_b_e.T * np.matrix(b)
                    # import pdb; pdb.set_trace()
            sum_z = sum_z + p_outputs[z] * sum_o
        if (max_alpha_b_e.T * np.matrix(b) + epsilon) < (sum_z.T * np.matrix(b)):
            max_alpha_b_e = sum_z
            best_e = firm.edges[i_v][i_e]
    return (max_alpha_b_e, best_e)


def plot_val(val):
    # v_names=['left','center','right']
    # q_names=['NoObsNoSample', 'Sample', 'Obs']
    for i_v in range(len(val)):
        for i_q in range(len(val[0])):
            fig = plt.figure(1)
            fig.add_subplot((len(val)), (len(val[0])), (i_v*len(val[0])+i_q))
            for i_alpha in range(val[i_v][i_q].alpha_mat.shape[1]):
                plt.plot(val[i_v][i_q].alpha_mat[:2, i_alpha])
                plt.text(val[i_v][i_q].belief_points[i_alpha][1],
                         val[i_v][i_q].alpha_mat[:, i_alpha].T*val[i_v][i_q].belief_points[i_alpha]+0.1,
                         str(val[i_v][i_q].best_edge[i_alpha]))
                # plt.title('Gamma(v='+v_names[i_v]+',q='+q_names[i_q]+')')
                plt.title('Gamma(v='+str(i_v)+',q='+str(i_q)+')')
                plt.xlabel('belief', horizontalalignment='right', x=1.0)
                plt.ylabel('Value')
                plt.ylim(-0.5, 1.5)
    plt.show()


if load:
    print "Loading Value Function"
    # fh = open('val.pkl', 'rb')
    # fh = open('val_' + sc + '_seed10_par.pkl', 'rb')
    fh = open('val_' + sc + '_seed10_par_newoutput.pkl', 'rb')
    val = pkl.load(fh)
    val_new = copy.deepcopy(val)
    fh.close()
else:
    # Hybrid Value Iteration
    val = [[Gamma(b_set) for i_q in range(dfsa.N)] for i_v in range(len(firm.nodes))]

    # Initialize Value Function to 1_(q_goal)
    for i_v in range(len(firm.nodes)):
        for i_q in range(dfsa.N):
            for i_b in range(len(val[i_v][i_q].belief_points)):
                if i_q in dfsa_final:
                    val[i_v][i_q].alpha_mat[:, i_b] = 1  #np.zeros([n_regs, 1])
                else:
                    val[i_v][i_q].alpha_mat[:, i_b] = 0  #np.zeros([n_regs, 1])
    n_cores = multiprocessing.cpu_count() - 1
    print "Running Value Iteration"
    val_new = copy.deepcopy(val)
    for i in range(10):
        print i
        for i_v in range(len(firm.nodes)):
            for i_q in range(dfsa.N):
                # Run backup for each belief point in parallel
                if parr:
                    results = Parallel(n_jobs=n_cores)(delayed(backup)(i_b, i_v, i_q, val)
                                                       for i_b in range(len(val[i_v][i_q].belief_points)))
                    for i_b in range(len(val[i_v][i_q].belief_points)):
                        val_new[i_v][i_q].alpha_mat[:, i_b] = results[i_b][0]
                        val_new[i_v][i_q].best_edge[i_b] = results[i_b][1]
                else:
                    for i_b in range(len(val[i_v][i_q].belief_points)):
                        alpha_new, best_e = backup(i_b, i_v, i_q, val)
                        val_new[i_v][i_q].alpha_mat[:, i_b] = alpha_new
                        val_new[i_v][i_q].best_edge[i_b] = best_e
        val = copy.deepcopy(val_new)
        fh = open('val_' + sc + '_seed10_par_newoutput.pkl', 'wb')
        pkl.dump(val, fh)
        fh.flush()
        fh.close()
        print "Val(v5,q0)"
        print val[5][0].alpha_mat
        print val[5][0].best_edge
        print "Val(v3,q0)"
        print val[3][0].alpha_mat
        print val[3][0].best_edge
        print "Val(v7,q0)"
        print val[7][0].alpha_mat
        print val[7][0].best_edge
if sc == 'toy':
    plot_val(val_new)


''' Execute the policy '''
b = env.b_prod_init
q = 0
v = 15
traj = []
for t in range(50):
    print "val = " + str(max(val[v][q].alpha_mat.T * b))
    # Get best edge
    i_best_alpha = np.argmax(val_new[v][q].alpha_mat.T * b)
    best_e = val_new[v][q].best_edge[i_best_alpha]
    # Simulate trajectory under edges
    edge_controller = firm.edge_controllers[v][firm.edges[v].index(best_e)]
    traj_e = edge_controller.simulate_trajectory(edge_controller.node_i)
    traj_n = firm.node_controllers[firm.nodes.index(edge_controller.node_j)].simulate_trajectory(traj_e[-1])
    # traj_i = [(b, i, q) for i in traj_e + traj_n]
    traj_i = traj_e + traj_n
    traj = traj + traj_i
    # Get q', v', q' and loop
    z = firm.get_outputs(traj_e + traj_n)
    (b_, o, i_o) = env.get_b_o(traj_n[-1].mean, b, x_e_true)
    v_ = best_e
    q_= None
    if regs[z][1]==1:
        if regs[z][2]=='obs' or regs[z][2]=='sample' and regs[z][0].contains(firm.nodes[v_].mean):
             q_ = np.argmax(dfsa.T(proplist[regs[z][2]])[q, :])
 # and regs[z][0].contains(firm.nodes[v_].mean)
    elif regs[z][1]>0:
        if (regs[z][2]=='obs' or regs[z][2]=='sample') and (env.x_e[i_o] & 2**env.reg_index[z] == 2**env.reg_index[z]) and regs[z][0].contains(firm.nodes[v_].mean):
             q_ = np.argmax(dfsa.T(proplist[regs[z][2]])[q, :])
    if q_ == None:
        q_ = q
    print "going from vertex " + str(v) + " to vertex " + str(v_) + " q = " + str(q) + " b = " + str(b)
    b = b_
    q = q_
    v = v_
    if q in dfsa_final:
        break
import pdb; pdb.set_trace()
firm.plot_traj(traj, 'blue')
plt.show()
