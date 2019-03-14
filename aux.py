import matplotlib.patches as patches
import polytope as pc
import matplotlib.pyplot as plt
import numpy as np
import math
import itertools
import warnings

def vertex_to_poly(v):
    data = pc.quickhull.quickhull(v)
    return pc.Polytope(data[0], data[1])


def plot_region(ax, poly, name, prob,
                color='red', alpha=0.5,
                hatch=False, fill=True):
    ax.add_patch(patches.Polygon(pc.extreme(poly), color=color, alpha=alpha, hatch=hatch, fill=fill))
    _, xc = pc.cheby_ball(poly)
    ax.text(xc[0] - 0.4, xc[1] - 0.43, '${}_{}$\n$p={}$'.format(name[0].upper(), name[1], prob))


def plot_nodes(nodes):
    assert isinstance(nodes, list)
    x = []
    y = []
    for node in nodes:
        x += np.ravel(node.mean)[0]
        y += np.ravel(node.mean)[1]

    plt.scatter(np.array(x), np.array(y), c="g")


def simulate(spath, regs, time_n=100,
             fig=None,
             obs_action=True):
    """
    Give a simulation of the current policy starting from the initial belief points

    :param time_n: total simulation time
    :type regs: Regions dictionary
    :param obs_action: obs_action true or false
    :param fig: figure handle (optional)
    :param spath: the specification road map
    :type spath: Spec_Spaths
    :return: trajectory, v_list, vals, act_list
    """
    if fig:
        pass
    else:
        fig = plt.figure()

    b = spath.env.b_prod_init  # initial belief points
    if len(spath.init) > 1:
        warnings.warn("Warning only first initial state is simulated")

    (q, v) = spath.init[0]

    # initialize
    traj = []
    v_list = [v]
    act_list = []
    vals = []
    vals_obs = []
    obs_list = []
    b_list = [b]

    q_ = None
    b_ = None
    for t in range(time_n):

        print(t)
        # Get best edge
        alpha_new, best_e, opt = spath._back_up_vert(q, v, b) # chose next node
        print("best_E", best_e)
        alpha_new, best_obs, opt_obs = spath._back_up_obs(q, best_e, b) # chose next observation
        if best_obs is None:
            #opt_obs = opt
            print(q, best_e)
            print(spath.node[(q, best_e)])
            print(spath.env.regs[spath.node[(q, best_e)]['reg']])
            best_obs =  -1 * (spath.env.regs.keys().index(spath.node[(q, best_e)]['reg'])+ 1)
            print("best ob",best_obs)

        vals += [opt]
        vals_obs += [opt_obs]


        # Simulate trajectory under edges
        edge_controller = spath.firm.edge_controllers[(v, best_e)]
        traj_e = edge_controller.simulate_trajectory(edge_controller.node_i)
        traj_n = spath.firm.node_controllers[edge_controller.node_j].simulate_trajectory(traj_e[-1])
        # traj_i = [(b, i, q) for i in traj_e + traj_n]
        traj_i = traj_e + traj_n
        traj = traj + traj_i
        # Get q', v', q' and loop
        z = spath.firm.get_output_traj(traj_e + traj_n)
        v_ = best_e
        act_list += [(best_e,best_obs)]
        if obs_action is False:
            print("TODO: Implement backup without obs_action")
        else:
            if regs[z][2] is 'null':
                b_ = b
                q_ = q
            elif isinstance(regs[z][2], str): # is not 'null' or regs[z][2] is 'sample1' or regs[z][2] is 'sample2':
                # line below is old code
                #elif regs[z][2] is 'obs' or regs[z][2] is 'sample1' or regs[z][2] is 'sample2':
                q_ = None
                b_ = None
                # if region is known
                if regs[z][1] == 1 or regs[z][1] == 0:
                    # if label is true
                    if regs[z][1] == 1:
                        q_ = spath.fsa.next_states_of_fsa(q, regs[z][2])
                        b_ = b
                # else update belief by simulating an observation
                else:
                    (b_, o, i_o) = spath.env.get_b_o_reg(b, spath.env.regs[z][3], z)
                    # if true label is true then do transition in DFA
                    # We are assuming that we get zero false rate when we pass through the region

                    if regs[z][3] is 1:
                        q_ = spath.fsa.next_states_of_fsa(q, (spath.env.regs[z][2],))
                if q_ is None:
                    q_ = q
                if b_ is None:
                    b_ = b
                print "Observing " + str(regs[z][3]) + " for " + z + " at vertex" + str(v_) + " q = " + str(q)
                obs_list += [(z, regs[z][3])]
        print("going from vertex " + str(v) + " to vertex " + str(v_) + " q = " + str(q_))

        b = b_  # new value becomes old value
        q = q_  # new value becomes old value
        v = v_  # new value becomes old value
        v_list += [v]
        if regs[z][2] is 'null' and best_obs < 0:
            reg_key = spath.env.regs.keys()[-1 * (best_obs + 1)]
            (b_, o, i_o) = spath.env.get_b_o_reg(b, spath.env.regs[reg_key][3], reg_key, v.mean)
            b = b_
            #act_list += ["Observing = " + reg_key]
            print "Observing " + str(i_o) + " for " + z + " at vertex" + str(v) + " q = " + str(q)
            obs_list += [(reg_key,i_o)]
        b = b_  # new value becomes old value

        b_list += [b]

        if isinstance(q,list):
            q = q[0]
        if not spath.active.get((q,v), False):

            print("opt_list", vals)
            print("act_list", act_list)

            break


    dist_list = []
    for n in v_list:
        if dist_list == []:
            dist_list += [0]
            n_last = n
            continue
        dist_list += [math.sqrt((n.mean[0] - n_last.mean[0]) ** 2 + (n.mean[1] - n_last.mean[1]) ** 2)]

        n_last = n

    vals += [1]
    vals_obs += [1]

    type = [ob[0] for ob in obs_list]
    d_2 = sum([[i,i] for i in np.cumsum(dist_list)],[])
    v_2 = sum([[z,j] for z,j in itertools.izip(vals,vals_obs)],[])

    d_2.pop(0)
    v_2.pop()

    plt.plot(d_2,v_2, marker='*')
    for i,d in enumerate(np.cumsum(dist_list)):
        if i <=0:
            continue
        plt.text(d+0.2, vals[i] + 0.06, type[i-1], fontsize=10)
        if obs_list[i-1][1] == 1:
            plt.scatter(d, vals[i] + 0.06, marker='^',color='b')
        else:
            plt.scatter(d, vals[i] + 0.06, marker='v',color='b')

    plt.title('Satisfaction probability')
    plt.xlabel('distance travelled')
    plt.ylabel('probability')
    plt.ylim(0, 1.2)



    return v_list, v_2,d_2, act_list,obs_list, b_list,dist_list


def plt_belief(b_list,env,dist=None):
    len_b = len(b_list[0]) # check number of unknown beliefs
    x_e = range(len_b)
    l_beliefs = int(math.log(len_b,2))
    print("number of unknown beliefs", l_beliefs)
    tr = 1
    b_identifiers = []
    for k in range(l_beliefs) :
        b_prod = np.matrix(np.zeros([len_b, 1]))
        for i in range(len_b):
            b_prod[i, 0] = 1
            for j in range(l_beliefs):
                if j == k:
                    if x_e[i] & 2 ** j == 2 ** j:
                        b_prod[i, 0] = b_prod[i, 0] * tr
                    else:
                        b_prod[i, 0] = b_prod[i, 0] * (1 - tr)
        b_identifiers += [b_prod] # the matrices that can be used to sum over the beliefs
    out = []
    for i,key in enumerate(env.regs):
        bel=[]
        for b in b_list:
            bel += [b.T * b_identifiers[i]]
        out += [np.concatenate(bel)]


    for b, re in zip(out, env.regs.keys()):
        if dist is not None:
            plt.plot(np.array(dist),np.array(b), label=re, marker='*')
        else:
            plt.plot(b, label=re, marker='*')


    plt.title('Belief state per region')
    plt.xlabel('distance travelled')
    plt.ylabel('probability')
    plt.ylim(0, 1.2)
    plt.legend()

    return out, env.regs.keys()