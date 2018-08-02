import best.rss18_functions as rf
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from best.hVI_fsrm import SPaths
from hVI_fsrm import spec_Spaths
import networkx as nx
if __name__ == '__main__':
    from best.ltl import formula_to_mdp
    from best.hVI_models import State_Space, Det_SI_Model
    from best.hVI_types import Env, Gamma
    import best.rss18_functions as rf
    from best.hVI_config import sc, load, parr, obs_action, epsilon, rand_seed
    import numpy as np
    from collections import OrderedDict
    import matplotlib.pyplot as plt
    import copy
    import cPickle as pkl
    import random
    from os import fork
    from joblib import Parallel, delayed
    import multiprocessing
    from itertools import product
    import time

    from polytope import *

    print "Setting Up Scenario"

    # Define Regions
    # Regs have the same format as RSS code. Regs that are added first have a higher priority
    #### Wall region
    print("Started wall case")
    regs = OrderedDict()
    p1 = rf.vertex_to_poly(np.array([[2, 0], [3, 0], [3, 1], [2, 1]]))
    regs['r1'] = (p1, 1, 'obs')
    p2 = rf.vertex_to_poly(np.array([[2, 1], [3, 1], [3, 2], [2, 2]]))
    regs['r2'] = (p2, 1, 'obs')
    p3 = rf.vertex_to_poly(np.array([[2, 2], [3, 2], [3, 3], [2, 3]]))
    regs['r3'] = (p3, 1, 'obs')
    p4 = rf.vertex_to_poly(np.array([[2, 3], [3, 3], [3, 4], [2, 4]]))
    regs['r4'] = (p4, .5, 'obs', 1)
    p5 = rf.vertex_to_poly(np.array([[2, 4], [3, 4], [3, 5], [2, 5]]))
    regs['r5'] = (p5, .3, 'obs', 1)

    p1 = rf.vertex_to_poly(np.array([[2, 0], [3, 0], [3, -1], [2, -1]]))
    regs['r6'] = (p1, 1, 'obs')
    p2 = rf.vertex_to_poly(np.array([[2, -1], [3, -1], [3, -2], [2, -2]]))
    regs['r7'] = (p2, 1, 'obs')
    p3 = rf.vertex_to_poly(np.array([[2, -2], [3, -2], [3, -3], [2, -3]]))
    regs['r8'] = (p3, 1, 'obs')
    p4 = rf.vertex_to_poly(np.array([[2, -3], [3, -3], [3, -4], [2, -4]]))
    regs['r9'] = (p4, .2, 'obs', 0)
    p5 = rf.vertex_to_poly(np.array([[2, -4], [3, -4], [3, -5], [2, -5]]))
    regs['r10'] = (p5, .7, 'obs', 0)

    a1 = rf.vertex_to_poly(np.array([[4, 0], [5, 0], [5, 1], [4, 1]]))
    regs['a1'] = (a1, 0.9, 'sample', 1)

    output_color = {'r1': 'red', 'r2': 'red', 'r3': 'red', 'r4': 'red', 'r5': 'red', 'r6': 'red', 'r7': 'red',
                    'r8': 'red', 'r9': 'red', 'r10': 'red',
                    'a1': 'green', 'null': 'white'}
    # Define Null regions with bounds of the space for null output with lowest priority
    p = rf.vertex_to_poly(np.array([[-3, -5], [-3, 5], [5, -5], [5, 5]]))
    regs['null'] = (p, 1.0, 'null')

    # Construct belief-MDP for env
    env = Env(regs)
    print(env)
    # belief set used for PBVI =>>> unclear what this is

    # fig = plt.figure(0)
    # ax = fig.add_subplot(111, aspect='equal')
    # l, u = bounding_box(p)
    # ax.set_xlim(l[0], u[0])
    # ax.set_ylim(l[1], u[1])
    # for (name, info) in regs.iteritems():
    #     hatch = False
    #     fill = True
    #     if name is not 'null':
    #         rf.plot_region(ax, info[0], name, info[1], output_color[name], hatch=hatch, fill=fill)

    # Construct and Visualize FIRM
    ''' Configuration Parameters '''
    random.seed(rand_seed)
    np.random.seed(rand_seed)

    print ''' Setup Motion and Observation Models '''
    # Define Motion and Observation Model
    Wx = np.eye(2)
    Wu = np.eye(2)
    r2_bs = State_Space([-5, -5], [5, 5])
    motion_model = Det_SI_Model(0.1)

    print " Constructing FIRM"
    fig = plt.figure(figsize=(14, 14), dpi=80, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(111, aspect='equal')
    firm = SPaths(r2_bs, motion_model, Wx, Wu, regs, output_color, ax)
    firm.make_nodes_edges(40, 3, init=np.array([[-4.5],[0]]))
    t1 = time.clock()
    firm.compute_output_prob()
    t2 = time.clock()
    firm.plot(ax)
    # plt.show()

    print(t2-t1)

    # Create DFA
    formula = '! obs U sample'

    prod_ = spec_Spaths(firm, formula,env)




    def plot_val(val):
        # v_names=['left','center','right']
        q_names=['NoObsNoSample', 'Sample', 'Obs']
        for i_v in range(len(val)):
            for i_q in range(len(val[0])):
                fig = plt.figure(1)
                fig.add_subplot((len(val)), (len(val[0])), (i_v*len(val[0])+i_q))
                for i_alpha in range(val[i_v][i_q].alpha_mat.shape[1]):
                    plt.plot(val[i_v][i_q].alpha_mat[:2, i_alpha])
                    plt.plot(val[i_v][i_q].b_prod_points[i_alpha][1],
                             val[i_v][i_q].alpha_mat[:, i_alpha].T*val[i_v][i_q].b_prod_points[i_alpha],
                             'ro')
                    plt.text(val[i_v][i_q].b_prod_points[i_alpha][1],
                             val[i_v][i_q].alpha_mat[:, i_alpha].T*val[i_v][i_q].b_prod_points[i_alpha]+0.1,
                             str(val[i_v][i_q].best_edge[i_alpha]))
                    # plt.title('Gamma(v='+v_names[i_v]+',q='+q_names[i_q]+')')
                    plt.title('Gamma(v='+str(i_v)+',q='+q_names[i_q]+')')
                    plt.xlabel('belief', horizontalalignment='right', x=1.0)
                    plt.ylabel('Value')
                    plt.ylim(-0.5, 1.5)
        # plt.show()


    not_converged = True
    i = 1
    opts =dict()
    while not_converged:
        print('iteration', i)
        not_converged = prod_.full_back_up(opts)
        i += 1



    prod_.plot_node([],'r4')
    ''' Execute the policy '''
    b = prod_.env.b_prod_init
    print b
    (q,key) = prod_.fsa.init.items()[0]
    (q,v) = (q,0)
    # v = 0
    traj = []
    v_list =[v]
    act_list =[]
    vals = []
    for t in range(150):
        # import pdb; pdb.set_trace()
        print((q,v))
        print(prod_.val[(q,v)].alpha_mat.T * np.array(b))
        #print "val = " + str(max(prod_.val[(q,v)].alpha_mat.T * b))
        vals += [str(max(prod_.val[(q,v)].alpha_mat.T * b))]
        # Get best edge
        # alpha_new, best_e, importance = backup_with_obs_action(i_b, v, q, val)
        # print("best action = ",best_e)

        i_best_alpha = np.argmax(prod_.val[(q,v)].alpha_mat.T * b)
        best_e = prod_.val[(q,v)].best_edge[i_best_alpha]

        if obs_action is True and best_e < 0:
            reg_key = prod_.env.regs.keys()[-1 * (best_e + 1)]
            (b_, o, i_o) = env.get_b_o_reg(b, env.regs[reg_key][3], reg_key, prod_.firm.nodes[v].mean)
            b = b_
            act_list += ["Observing = " +reg_key ]
            print "Observing " +reg_key + " at vertex" + str(v) + " q = " + str(q) + " b_ = " + str(b_)
            continue
        # Simulate trajectory under edges
        edge_controller = firm.edge_controllers[v][firm.edges[v].index(best_e)]
        traj_e = edge_controller.simulate_trajectory(edge_controller.node_i)
        traj_n = firm.node_controllers[firm.nodes.index(edge_controller.node_j)].simulate_trajectory(traj_e[-1])
        # traj_i = [(b, i, q) for i in traj_e + traj_n]
        traj_i = traj_e + traj_n
        traj = traj + traj_i
        # Get q', v', q' and loop
        z = firm.get_output_traj(traj_e + traj_n)
        v_ = best_e
        act_list += [best_e]

        if obs_action is False:
            print "TODO: Implement backup without obs_action"
        else:
            if regs[z][2] is 'null':
                b_ = b
                q_ = q
            elif regs[z][2] is 'obs' or regs[z][2] is 'sample':
                q_ = None
                b_ = None
                # if region is known
                if regs[z][1] == 1 or regs[z][1] == 0:
                    # if label is true
                    if regs[z][1] == 1:
                        q_ = np.argmax(dfsa.T(proplist[regs[z][2]])[q, :])
                        b_ = b
                # else update belief by simulating an observation
                else:
                    (b_, o, i_o) = env.get_b_o_reg(b, env.regs[z][3], z)
                    # if true label is true then do transition in DFA
                    # We are assuming that we get zero false rate when we pass through the region
                    if regs[z][3] is 1:
                        q_ = np.argmax(dfsa.T(proplist[regs[z][2]])[q, :])
                if q_ is None:
                    q_ = q
                if b_ is None:
                    b_ = b
        print "going from vertex " + str(v) + " to vertex " + str(v_) + " q = " + str(q) + " b = " + str(b)
        b = b_
        q = q_
        v = v_
        v_list += [v]
        if q in dfsa_final:
            break
    print(v_list)
    print(vals)
    print(act_list)
    firm.plot_traj(traj, 'green')
    plt.show()
