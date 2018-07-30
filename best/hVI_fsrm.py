'''
FSRM = Feedback State RoadMap
This file gives the class for the generation of a sampling based path planner
that can be used together with the HVI tools.
'''
from best.fsa import Fsa
import networkx as nx
from best.mdp import MDP
from best.hVI_models import Det_SI_Model, State_Space, State
import best.rss18_functions as rf
import time
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.linalg import solve_discrete_are as dare
import itertools as it
from collections import OrderedDict
from collections import *
from tulip.transys.labeled_graphs import LabeledDiGraph
from best.ltl import formula_to_mdp
from itertools import product
import random
from best.hVI_types import Env, Gamma
import logging
logger = logging.getLogger(__name__)

class SPaths(object):
    # belief_space, motion_model: Refer to classes in models.py
    # Wx = quadratic cost matrix of state used in LQR
    # Wu = quadratic cost matrix of input used in LQR
    # regs = same as RSS definition
    # regs_outputs = Mapping from regs info[2] to integer of output; e.g. regs_output = {'blue':0, 'green':1, 'red':2}
    # output_color = e.g. output_color = {-1:'black', 0:'blue', 1:'green', 2:'red'}
    # ax = handle for plotting stuff, can be generated as: fig = plt.figure(0); ax = fig.add_subplot(111, aspect='equal')
    def __init__(self, state_space, motion_model, Wx, Wu, regs, output_color, ax):
        self.state_space = state_space
        # verify that the motion model is deterministic

        if not isinstance(motion_model, Det_SI_Model):
            raise TypeError
        self.motion_model = motion_model
        self.obs_model = None
        self.regs = regs
        assert np.all(np.linalg.eigvals(Wx) > 0)
        assert np.all(np.linalg.eigvals(Wu) > 0)
        self.Wx = Wx
        self.Wu = Wu
        self.output_color = output_color
        self.ax = ax
        self.nodes = []
        self.node_controllers = []
        self.edges = OrderedDict()  # key=node_id and value=list of node_ids of neighbors
        self.edge_controllers = OrderedDict()
        self.edge_output_prob = OrderedDict()
        self.T_list = None
        self.Tz_list = None

    def make_nodes_edges(self, number, edges, init=None):
        t = [time.clock()]
        self.sample_nodes(number, init=init)
        t += [time.clock()]

        self.make_edges(edges)
        t += [time.clock()]

        self.n_particles = 1
        print(np.diff(t))

    def sample_nodes(self, n_nodes, means=list(), append=False, init= None):
        # ''' Sample nodes in belief space and also generate node_controllers '''
        # n_nodes = number of nodes to sample in graph
        # append = False erases all previous nodes whereas True adds more nodes to existing graph

        # TODO: Implement append to sample nodes incrementally
        if not means and n_nodes < len(self.regs):
            raise ValueError('Number of samples cannot be less than n_regs')


        if append is False:
            self.nodes = []  # clear previous nodes/edges
            self.edges = OrderedDict()
            self.node_controllers = []
            self.edge_controllers =OrderedDict()


        for i in range(n_nodes):
            # if i == 4:
            #     import pdb; pdb.set_trace()
            # Sample Mean
            if means:
                if len(means) is not n_nodes:
                   raise ValueError('means does not have n_nodes values')
                node = self.state_space.new_state(means[i])
            else:
                # add sample to every region that is not an obstacle
                if i == 0 :
                    if isinstance(init, np.ndarray):
                        node = State(init)
                    else:
                        resample = True
                        while resample:
                            resample = False
                            node = self.state_space.sample_new_state()

                            for key, value in self.regs.iteritems():
                                if value[2] is 'obs' and value[0].contains(node.mean):
                                    resample = True

                elif i < len(self.regs)+1 and ((self.regs[self.regs.keys()[i-1]][2] is not 'obs') or (self.regs[self.regs.keys()[i-1]][1]<1)):
                        node = self.state_space.sample_new_state_in_reg(self.regs[self.regs.keys()[i-1]][0])
                else:
                    # Implemented rejection sampling to avoid nodes in obs => why would you do that?
                    # you want to avoid samples in regions that we know to be obstacles,
                    # but if they could be obstacles and it is not sure, then it makes more sense to keep samples in them
                    resample = True
                    while resample:
                        resample = False
                        node = self.state_space.sample_new_state()

                        for key, value in self.regs.iteritems():
                            if value[2] is 'obs' and value[0].contains(node.mean):
                                resample = True

            # Set Co-variance
            A = self.motion_model.getA(node)


            self.nodes.append(node)
            self.node_controllers.append(
                Node_Controller(self.motion_model, self.obs_model,
                                node, self.Wx, self.Wu,
                                self.state_space))
        # TODO:''' Generate one node in each region '''
        # j=0
        # for key in self.regs:
        #   x_low = self.regs[key][0].bounding_box[0].ravel()
        #   x_up = self.regs[key][0].bounding_box[1].ravel()
        #   for i in range(len(self.x_low)):
        #     self.nodes[n_nodes+j,i] = x_low[i] + (x_up[i] - x_low[i])*np.random.rand(1).ravel()
        #   j=j+1

    ''' Construct edges for self.nodes within distance dist and generate edge controllers '''
    # dist = distance (on mean) threshold for neighbors in PRM
    def make_edges(self, dist):
        self.max_actions = 1  # '''used to construct T'''
        # Can make more efficient
        t = [time.clock()]
        visited=dict()
        for i in range(len(self.nodes)):
            neigh = []
            edge_controllers = []
            ti = []
            for j in range(len(self.nodes)):
                if i == j:
                    continue
                add_edge = False
                if frozenset({i,j}) in visited:
                    add_edge = visited[frozenset({i,j})]
                else:

                    dist_nodes = self.state_space.distance_mean(self.nodes[i], self.nodes[j])
                    add_edge = dist_nodes < dist
                    if add_edge:

                        # check whether the kripke condition is satisfied
                        output = set(self.get_outputs(self.nodes[i], self.nodes[j]))
                        output_ends = set(self.get_outputs(self.nodes[i]))
                        output_ends |= set(self.get_outputs(self.nodes[j]))


                        if not(output.issubset(output_ends)):
                            print(('remove edge',{i,j}))
                            print(output)
                            add_edge = False

                    visited[frozenset({i, j})] = add_edge

                if add_edge:
                    neigh.append(j)
                    edge_controllers.append(Edge_Controller(self.motion_model,self.obs_model,self.nodes[i],self.nodes[j],self.Wx,self.Wu,self.state_space))
            if len(neigh) > self.max_actions:
                self.max_actions = len(neigh)



            self.edges[i] = neigh
            self.edge_controllers[i] = edge_controllers

    def intersect(self,box,src,dest):
        diff = dest - src
        ranges = np.append(*box.bounding_box, axis=1)

        low, high = box.bounding_box

        u = np.zeros((2,))
        v = np.ones((2,))

        if abs(diff[0]) < np.finfo(float).eps: # constant along the x-axis
            if not (ranges[0, 0] <= src[0] <= ranges[0, 1]):
                return False
        else:
            u[0] = max(min((low[0] - src[0])/diff[0],(high[0] - src[0])/diff[0]),0)
            v[0] = min(max((low[0] - src[0])/diff[0],(high[0] - src[0])/diff[0]),1)

        if abs(diff[1]) < np.finfo(float).eps: # constant along the y-axis
            if not (ranges[1, 0] <= src[1] <= ranges[1, 1]):

                return False
        else:
            u[1] = max(min((low[1] - src[1])/diff[1],(high[1] - src[1])/diff[1]),0)
            v[1] = min(max((low[1] - src[1])/diff[1],(high[1] - src[1])/diff[1]),1)
        assert (v<=1).all()
        assert (u>=0).all()
        return np.max(u) <= np.min(v)

    ''' Compute the probability over output set for each edge in the FIRM graph '''
    def compute_output_prob(self):
        for (node, edge_controllers) in self.edge_controllers.iteritems():
            output_prob_edges = []
            for edge_controller in edge_controllers:
                output = self.get_outputs(edge_controller.node_i)
                p_out = OrderedDict([(key, 0) for key, value in self.regs.iteritems()])
                for out in output:
                    p_out[out] = 1
                output_prob_edges.append(p_out)
            self.edge_output_prob[node] = output_prob_edges

    def get_outputs(self, src, dest=None):
        output = ['null']
        start = (src.mean if isinstance(src, State) else src)
        if dest:
            end = (dest.mean if isinstance(dest, State) else dest)
            for (name, info) in self.regs.iteritems():
                if name == 'null':
                    continue
                box = info[0]
                if self.intersect(box, start, end):
                    output += [name]
                    # if self.ax is not None:
                    #     self.plot_traj([src] + [dest], self.output_color[name])
            return output
        else:  # this is if dest is None
            for (name, info) in self.regs.iteritems():
                if name == 'null':
                    continue
                box = info[0]
                if box.contains(start):
                    output = [name]
            return output

    ''' Returns a set of outputs generated by the trajectory.
    For now it returns only one output with the lowest position in the dictionary,
    even if multiple outputs are generated '''
    # traj = list of belief_state(s)
    # [R] output = key of regs
    def get_output_traj(self, traj):
        output = 'null'
        for belief in traj:
            for (name, info) in self.regs.iteritems():
                poly = info[0]
                if poly.contains(belief.mean):
                    if self.regs.keys().index(output) > self.regs.keys().index(name):
                        output = name
        return output

    ''' Plot a trajectory in belief space'''
    # traj = list of belief_state(s)
    # color = color for argument of plot
    def plot_traj(self, traj, color):
        for i in range(len(traj)-1):
            if isinstance(traj[i], State):
                try:
                    x = [np.ravel(traj[i].mean)[0], np.ravel(traj[i+1].mean)[0]]
                    y = [np.ravel(traj[i].mean)[1], np.ravel(traj[i+1].mean)[1]]
                except:
                    x = [np.ravel(traj[i].mean)[0], np.ravel(traj[i+1])[0]]
                    y = [np.ravel(traj[i].mean)[1], np.ravel(traj[i+1])[1]]

            else:
                x = [np.ravel(traj[i])[0], np.ravel(traj[i+1])[0]]
                y = [np.ravel(traj[i])[1], np.ravel(traj[i+1])[1]]
            # if color == 'white':
            #     color = 'black'
            self.ax.plot(x, y, color, ms=20, linewidth=3.0)

    ''' Plot the FIRM graph '''
    # ax = handle to plot
    def plot(self, ax):
        for i in range(len(self.nodes)):
            try:
                neigh = self.edges[i]
            except KeyError:
                continue
            for j in neigh:
                x = [np.ravel(self.nodes[i].mean)[0], np.ravel(self.nodes[j].mean)[0]]
                y = [np.ravel(self.nodes[i].mean)[1], np.ravel(self.nodes[j].mean)[1]]
                ax.plot(x, y, 'b')
        scale = 3
        for i in range(len(self.nodes)):
            # ax.plot(self.nodes[i].mean[0], self.nodes[i].mean[1], 'go')
            from matplotlib.patches import Ellipse
            eigvalue, eigvec = np.linalg.eigh(self.nodes[i].cov[0:2, 0:2])
            if eigvalue[0] < eigvalue[1]:
                minor, major = 2 * np.sqrt(scale * eigvalue)
                alpha = np.arctan(eigvec[1, 1] / eigvec[0, 1])
            elif eigvalue[0] > eigvalue[1]:
                major, minor = 2 * np.sqrt(scale * eigvalue)
                alpha = np.arctan(eigvec[1, 0] / eigvec[0, 0])
            else:
                major, minor = 2 * np.sqrt(scale * eigvalue)
                alpha = 0
            ell = Ellipse(xy=(self.nodes[i].mean[0], self.nodes[i].mean[1]),
                          width=major, height=minor, angle=alpha)
            ell.set_facecolor('gray')
            ax.add_artist(ell)
            if i < 10:
                plt.text(np.ravel(self.nodes[i].mean)[0]-0.04, np.ravel(self.nodes[i].mean)[1]-0.05, str(i), color='black',backgroundcolor='grey')
            else:
                plt.text(np.ravel(self.nodes[i].mean)[0]-0.09, np.ravel(self.nodes[i].mean)[1]-0.05, str(i), color='black',backgroundcolor='grey')
        ax.set_xlim(self.state_space.x_low[0], self.state_space.x_up[0])
        ax.set_ylim(self.state_space.x_low[1], self.state_space.x_up[1])
        for (name, info) in self.regs.iteritems():
            hatch = False
            fill = True
            if name is not 'null':
                rf.plot_region(ax, info[0], name, info[1], self.output_color[name], hatch=hatch, fill=fill)
        # plt.show()

    ''' Construct T by treating index of neigh as action number
    TODO: Update this for Hybrid-PBVI MDP definition '''
    # [R] MDP = abstraction
    def abstract(self):
        nodes_start_list = [[] for i in range(self.max_actions)]
        nodes_end_list = [[] for i in range(self.max_actions)]
        vals_list = [[] for i in range(self.max_actions)]
        for i in range(self.nodes.shape[0]):
            neigh = self.edges[i]
            for j in range(len(neigh)):
                nodes_start_list[j].append(i)
                nodes_end_list[j].append(neigh[j])
                vals_list[j].append(1)
        self.T_list = []
        for i in range(self.max_actions):
            self.T_list.append(
                sp.coo_matrix((vals_list[i],
                              (nodes_start_list[i],
                               nodes_end_list[i])),
                              shape=(self.nodes.shape[0],
                                     self.nodes.shape[0])))
        output_fcn = lambda s: self.nodes[s]
        return MDP(self.T_list, output_name='xc', output_fcn=output_fcn)


class Node_Controller(object):
    ''' Consists of SLGR and SKF '''
    # belief_space, motion_model, obs_model: Refer to classes in models.py
    # Wx = quadratic cost matrix of state used in LQR
    # Wu = quadratic cost matrix of input used in LQR
    def __init__(self, motion_model, obs_model, node, Wx, Wu, state_space):
        self.motion_model = motion_model
        self.obs_model = obs_model # = None
        self.A = self.motion_model.getA(node)
        self.B = self.motion_model.getB(node)
        self.node = node
        self.Wx = Wx
        self.Wu = Wu
        self.state_space = state_space
        # set control gain
        #S = np.mat(dare(self.A, self.B, self.Wx, self.Wu))
        self.motion_model.getLS(Wx,Wu)
        self.Ls = self.motion_model.Ls #(self.B.T * S * self.B + self.Wu).I * self.B.T * S * self.A

    ''' Simulates trajectory form node_i to node_j '''
    # b0 = initial belief_state
    def simulate_trajectory(self, b0):
        traj = [b0]
        while not self.state_space.distance(traj[-1], self.node) < 0.1:
            ''' Get control, apply control, get observation and apply observation '''
            b = traj[-1]
            # Get control
            u_k = -self.Ls * (np.mat(b) - self.node.mean)

            # Apply control/predict (Esq. 75-76, 98-99)
            # x^-_k+1 = A * x^+_k + B * u; P^-_k+1 = A * P^+_k *A^T + G Q G^T
            bnew_pred = self.motion_model.evolve(b, u_k)

            traj.append(self.state_space.new_state(bnew_pred))
        return traj


class Edge_Controller(object):
    ''' Time varying LQG controller '''
    # belief_space, motion_model, obs_model: Refer to classes in models.py
    # Wx = quadratic cost matrix of state used in LQR
    # Wu = quadratic cost matrix of input used in LQR
    def __init__(self, motion_model, obs_model, node_i, node_j, Wx, Wu, state_space):
        self.motion_model = motion_model
        self.obs_model = obs_model # = None
        self.node_i = node_i
        self.node_j = node_j
        # self.Wx = Wx
        # self.Wu = Wu
        self.state_space = state_space
        [self.traj_d, self.u0] = self.motion_model.generate_desiredtraj_and_ffinput(node_i, node_j)

        self.N = len(self.traj_d)
        n_xdim = self.motion_model.getA(node_i).shape[1]
        n_udim = self.motion_model.getB(node_i).shape[1]
        # Generate feedback gains
        # S = np.empty((self.N + 1, n_xdim, n_xdim), dtype=np.float)
        self.L = np.empty((self.N, n_udim, n_xdim), dtype=np.float)
        # S[self.N, :, :] = self.Wx
        for t in it.count(self.N - 1, -1):
            #A, B, S_next = map(np.mat, [self.motion_model.getA(self.traj_d[t]),
            #                           self.motion_model.getB(self.traj_d[t]),
            #                           S[t + 1, :, :]])
            #L = (B.T * S_next * B + self.Wu).I * B.T * S_next * A

            self.L[t, :, :] = self.motion_model.Ls
            #S[t, :, :] = self.Wx + A.T * S_next * (A - B * L)
            if t == 0:
                break

    ''' Simulates trajectory starting from node_i to node_j '''
    def simulate_trajectory(self, b0):

        # traj = [self.node_i]
        traj = [b0]
        for t in range(self.N-1):
            b = traj[-1]

            u = -self.L[t, :, :] * (b.mean - self.traj_d[t]) + self.u0[t]
            bnew_pred = self.motion_model.evolve(b, u)
            # Update Belief
            traj.append(self.state_space.new_state(bnew_pred))
        return traj


class spec_Spaths(nx.MultiDiGraph):
    # includes the SPaths information
    def __init__(self,SPaths_object, formula,env):
        # type: (SPaths_object, formula, env)

        self.dfsa, self.dfsa_init, self.dfsa_final, self.proplist = formula_to_mdp(formula)
        self.fsa = Fsa()
        self.fsa.from_formula(formula)
        self.fsa.add_trap_state()
        # initialize with DFA and SPath object
        self.firm = SPaths_object
        self.env = env
        probs = [0.1, 0.2, 0.5, 0.8, .9]  # what is this?
        self.probs_list = [probs for i in range(env.n_unknown_regs)]
        self.b_reg_set = [env.get_reg_belief(list(i)) for i in product(*self.probs_list)]
        b_prod_set = [env.get_product_belief(list(i)) for i in product(*self.probs_list)]
        # True state of the regs used to simulate trajectories after policy is generated
        # x_e_true
        n = 50
        self.b_prod_set = random.sample(b_prod_set, n)
        self.epsilon = 10**-5

        self.val = dict()
        self.active = dict()

        # values will point to values of _*_label_def below


        types = [
            {'name': 'input',
             'values': {'null'}|set(self.fsa.props.keys()),
             'setter': True,
             'default': 'null'}]
        reg_names = [{'name': 'reg',
                     'values': {'null'}|set(self.env.regs.keys()),
                     'setter': True,
                     'default': 'null'}]

        super(spec_Spaths, self).__init__(state_label_types=reg_names , edge_label_types=types)
        self.sequence = self.create_prod()



    def _make_edge_to(self,from_pair,  node_pair, label, unvisited):
        if not node_pair in self.nodes():
            self.add_node(node_pair)
            #print('final' self.fsa.final)
            if  node_pair[0] in self.fsa.final:
                self.val[node_pair].alpha_mat[:, 0] = 1
                self.active[node_pair] = False
                print('adding edge to target')
                self.add_edge(node_pair, (-1,-1))

            else:

                unvisited += [node_pair]

        # TODO: add right labels for inputs
        self.add_edge(from_pair,node_pair,input=label)

        return unvisited


    def create_prod(self):
        # Add nodes based on which one can actually be reached
        unvisited = []
        for i_q in [state for (state, key) in self.fsa.init.items() if key == 1] :
            # add all initial dfa states and initial graph stats (v=0)
            self.add_node((i_q, 0))
            unvisited += [(i_q, 0)]

        # add a virtual final node () can be used for breath first searchs

            super(spec_Spaths, self).add_node((-1,-1))
            self.active[(-1,-1)] = False


        while len(unvisited)>0:
            (i_q,i_v) = unvisited.pop(0)
            # TODO Add edges start from initial node
            for v_next in self.firm.edges[i_v]:
                # TODO complete this with obs and region labels
                # compute output of that vertex
                list_labels = self.firm.get_outputs(self.firm.nodes[v_next])

                try:
                    list_labels.remove('null')
                except ValueError: pass

                for (orig_q_, q_next, label_dict) in self.fsa.g.out_edges(i_q, data=True):
                    if 0 in label_dict['input']:
                        unvisited = self._make_edge_to((i_q,i_v),(q_next, v_next), 'null', unvisited)
                    #print(list_labels)
                    if len(list_labels)>0 and self.fsa.bitmap_of_props((self.env.get_prop(list_labels[0]),)) in label_dict['input']:
                        #print(list_labels, self.fsa.bitmap_of_props(('sample',)), label_dict['input'])
                        # todo allow for more labels as input ('sample ^ obstacle')
                        unvisited = self._make_edge_to((i_q,i_v),(q_next, v_next), self.env.get_prop(list_labels[0]), unvisited)
                        # test edge
                        self.find_edge((i_q,i_v),self.env.get_prop(list_labels[0]))
        edges = nx.bfs_edges(self,(-1,-1), reverse=True)
        u_nodes = OrderedDict() # this dictionary will give the sequence of nodes to iterate over
        for u,v in edges:
            u_nodes[u] = True

        u_nodes[(-1,-1)] = False
        for u in u_nodes:
            if not self.active[u] :
                u_nodes[u] = False
        return u_nodes


    def add_node(self, n, attr_dict=None, check=True, **attr):
        # add node n to graph
        list_labels = self.firm.get_outputs(self.firm.nodes[n[1]])
        if len(list_labels) > 1:
            list_labels.remove('null')

        super(spec_Spaths, self).add_node(n, attr_dict=attr_dict, check=check, reg=list_labels[-1])
        self.val[n] = Gamma(self.b_prod_set, self.b_reg_set)  # added i_q,i_v
        self.active[n] = True

    def rem_node(self,n):
        # (n=node)=> remove node from graph
        super(spec_Spaths, self).remove_node(n)
        self.active.__delitem__(n)
        self.val.__delitem__(n)

    def find_edge(self, n, input, v=None):
        if v is None:
            for (n_,next_n,dict_input) in self.out_edges({n}, data='input') :
                if dict_input ==input:
                    return next_n
        for (n_,next_n,dict_input) in self.out_edges({n}, data='input') :
            if dict_input ==input and next_n[1] == v:

                return next_n[0]

        raise ValueError

    def full_back_up(self):
        t0 = time.clock()
        print('Do full back-up')
        for n in self.sequence:
            # do back up
            self.back_up(n[0],n[1])
        t1 = time.clock()
        print(t1-t0)
        return any(self.sequence.values())
        # boolean value giving whether or not the backups have converged

    def back_up(self, i_q, i_v, b=None):
        #print(i_q, i_v)

        if self.active[(i_q, i_v)] == False or self.sequence[(i_q, i_v)] == False:
            return

        # check first whether a backup is really needed
        # if all neighbors in the self graph have become inactive and
        # if it is false in the sequence then also this node can be set to false

        #print('compute for node {n}'.format(n= (i_q, i_v)))
        if isinstance(b,np.ndarray): # if no b is given then do it for all of them
            return self._back_up(i_q, i_v, b)
        else:
            alph_list = []
            i_b = 0
            # Get belief point from value function
            for b in self.val[(i_q, i_v)].b_prod_points:

                alpha_new, best_e, importance = self._back_up(i_q,i_v, b)
                alph_list += [alpha_new]
                self.val[(i_q, i_v)].best_edge[i_b] = best_e
                i_b += 1
            alpha_mat = np.matrix(np.unique(np.array(np.concatenate(alph_list, axis=1)),axis=1))
            try:
                self.sequence[(i_q, i_v)] = not np.allclose(alpha_mat,self.val[(i_q, i_v)].alpha_mat,rtol=1e-03, atol=1e-03)
            except:
                self.sequence[(i_q, i_v)] = True



            if self.sequence[(i_q, i_v)]:# you did change something, all neighbors could be affected
                for n_out in self.successors((i_q, i_v)):
                    if n_out in
                    self.sequence[(i_q, i_v)] = True

            self.val[(i_q, i_v)].alpha_mat = alpha_mat

            return


    def _back_up(self, i_q, i_v, b):




        epsilon = self.epsilon
        # Set max alpha and best edge to current max/best (need this to avoid invariant policies)
        # Find index of best alpha from gamma set
        index_alpha_init = np.argmax(self.val[(i_q, i_v)].alpha_mat.T * b)
        # Save best alpha vector
        max_alpha_b_e = self.val[(i_q, i_v)].alpha_mat[:, index_alpha_init]
        # Save edge corresponding to best alpha vector
        best_e = self.val[(i_q, i_v)].best_edge[index_alpha_init]
        nf = (i_q, i_v) # source node
        # Foreach edge action
        for v_e in self.firm.edges[i_v] :
            # Get probability of reaching goal vertex corresponding to current edge
            # p_reach_goal_node = firm.reach_goal_node_prob[i_v][i_e]
            p_reach_goal_node = 0.99  # TODO Get this from Petter's Barrier Certificate
            # next node if observing no label
            q_z_o = self.find_edge((i_q, i_v), 'null', v=v_e)
            n = (q_z_o, v_e)  # next node
            # Get goal vertex corresponding to edge
            z = self.node[n]['reg']


            # TODO: Remove this hardcoding
            # If output is null or region is known
            if z is 'null':


                # Get gamma set corresponding to v and q after transition
                gamma_e = np.matrix(self.val[n].alpha_mat)
                # Get index of best alpha in the gamma set
                index = np.argmax(gamma_e.T * b)
                # Get new alpha vector by scaling down best alpha by prob of reaching goal node
                alpha_new = p_reach_goal_node * gamma_e[:, index]
            else:
                alpha_new = 0
                O = self.env.get_O_reg_prob(z)
                for i_o in range(2):
                    q_z_o = None
                    # if we end up in obstacle/sample region and also observe obstacle/sample
                    if i_o == 1:
                        # Transition to new q
                        q_z_o = self.find_edge(nf,self.env._regs[z][2],v=v_e)
                        #q_z_o = np.argmax(self.dfsa.T(self.proplist[self.env._regs[z][2]])[i_q, :])
                    else:
                        # new q = current q
                        q_z_o = self.find_edge(nf,'null',v=v_e)
                    gamma_e = np.diag(np.ravel(O[i_o, :])) * np.matrix(self.val[(q_z_o,v_e)].alpha_mat)
                    index = np.argmax(gamma_e.T * b)
                    alpha_new = alpha_new + p_reach_goal_node * gamma_e[:, index]
            # Update max_alpha and best_edge if this has a greater value
            if (max_alpha_b_e.T * np.matrix(b) + epsilon) < (alpha_new.T * np.matrix(b)):
                max_alpha_b_e = alpha_new
                best_e = v_e
            # if (i_v==42 and i_q==0 and i==7):  # for debugging only
            # print "obs = " + str(i_o) + "q_z = " + str(q_z)
            # print sum_z.T * b
            # print max_alpha_b_e.T * np.matrix(b)
            # import pdb; pdb.set_trace()

        # Foreach obs action (iterate through every region that we can observe)
        for key, info in  self.env.regs.iteritems():
            # Get observation matrix as defined in the paper
            O = self.env.get_O_reg_prob(key, self.firm.nodes[i_v].mean)
            # Initialize sum over observations to zero
            sum_o = np.zeros([2 ** self.env.n_unknown_regs, 1])
            # Iterate over possible observations/Labels (True, False)
            for i_o in range(2):
                # Get new Gamma set
                gamma_o_v = np.diag(np.ravel(O[i_o, :])) * np.matrix(self.val[(i_q, i_v)].alpha_mat)
                # Find index of best alpha in the new Gamma set
                index = np.argmax(gamma_o_v.T * b)
                # Add the best alpha to the summation
                sum_o = sum_o + gamma_o_v[:, index]
            # Check if new alpha has a greater value
            if (max_alpha_b_e.T * np.matrix(b) + epsilon) < (sum_o.T * np.matrix(b)):
                # Update the max_alpha and best_edge
                max_alpha_b_e = sum_o
                best_e = -1 * (self.env.regs.keys().index(key) + 1)  # region 0 will map to edge -1

        # Sanity check that alpha <= 1
        if not (max_alpha_b_e <= 1).all():
            print('warning ' , max_alpha_b_e, best_e)
            assert False

        return (max_alpha_b_e, best_e, (max_alpha_b_e > 0).any())


    #def plot_node(self,node, region ):
