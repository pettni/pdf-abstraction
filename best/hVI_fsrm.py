'''
FSRM = Feedback State RoadMap
This file gives the class for the generation of a sampling based path planner
that can be used together with the HVI tools.
'''
from best.fsa import Fsa
from copy import copy
import networkx as nx
import itertools
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
    '''
    Construct Roadmap object with
    belief_space, motion_model: Refer to classes in models.py
    # Wx = quadratic cost matrix of state used in LQR
    # Wu = quadratic cost matrix of input used in LQR
    # regs = same as RSS definition
    # regs_outputs = Mapping from regs info[2] to integer of output; e.g. regs_output = {'blue':0, 'green':1, 'red':2}
    # output_color = e.g. output_color = {-1:'black', 0:'blue', 1:'green', 2:'red'}
    # ax = handle for plotting stuff, can be generated as: fig = plt.figure(0); ax = fig.add_subplot(111, aspect='equal')
    '''
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

    def make_nodes_edges(self, number, edges_dist, init=None):
        t = [time.clock()]
        self.sample_nodes(number, init=init)
        t += [time.clock()]

        self.make_edges(edges_dist)
        t += [time.clock()]

        self.n_particles = 1
        print(np.diff(t))

    def sample_nodes(self, n_nodes, means=list(), append=False, init=None, min_dist=1.7):
        ''' Sample nodes in belief space and also generate node_controllers
         :param n_nodes  = number of nodes to sample in graph
         :param append = False erases all previous nodes whereas True adds more nodes to existing graph
         :param min_dist = Samples within min_dist range will be rejected unless they produce non-NULL outputs
                   can be set to 0.0 to disable pruning

         '''

        # TODO: Implement append to sample nodes incrementally
        if not means and n_nodes < len(self.regs):
            raise ValueError('Number of samples cannot be less than n_regs')

        # make a list of nodes
        nodes = []
        if append is False:
            self.nodes = []  # clear previous nodes/edges
            self.edges = OrderedDict()
            self.node_controllers = []
            self.edge_controllers =OrderedDict()



        if means: #=> if you already have a precomputed node to add use that one first.
            pass
        elif not append: # first use init
            if isinstance(init, np.ndarray):
                nodes += [State(init)] # add first mean for graph node
            for reg in self.regs:
                if (self.regs[reg][2] is not 'obs') or (self.regs[reg][1]<1):
                    nodes += [self.state_space.sample_new_state_in_reg(self.regs[reg][0])]

        while means:
            nodes += [self.state_space.new_state(means.pop(0))]

        added_nodes = []
        while n_nodes>0:
            print(n_nodes)
            if nodes:
                node = nodes.pop(0) # keep taking first node until empty
            else:
                # Implemented rejection sampling to avoid nodes in obs => why would you do that?
                # you want to avoid samples in regions that we know to be obstacles,
                # but if they could be obstacles and it is not sure, then it makes more sense to keep samples in them
                node = self.state_space.sample_new_state()
                resample = False
                if any(map(lambda value: value[2] is 'obs' and value[0].contains(node.mean), self.regs.values())):
                    continue  # now you dont implement this sample

                # # Reject sample if it is too close any previous sample and has null outputs
                # for node_j in added_nodes:
                #     dist_nodes = self.state_space.distance_mean(node, node_j)
                #     if dist_nodes < min_dist:
                #         output_edge = set(self.get_outputs(node, node_j))
                #         output_start = set(self.get_outputs(node))
                #         output_end = set(self.get_outputs(node_j))
                #
                #         if output_start == output_end and output_start.issubset(output_edge):
                #             print "discard this sample"
                #             print "edge = " + str(output_edge)
                #             print "start = " + str(output_start)
                #             print "end = " + str(output_end)
                #             resample = True
                #             break
                # if resample == True :
                #     continue
                # This does NOT work, it makes everything break!!



            # Set Co-variance
            self.nodes.append(node)
            self.node_controllers.append(
                Node_Controller(self.motion_model, self.obs_model,
                                node, self.Wx, self.Wu,
                                self.state_space))

            added_nodes += [node]
            n_nodes += -1
        return added_nodes

    def make_edges(self, dist,nodes=list()):
        '''
        Construct edges for self.nodes within distance dist and generate edge controllers
        :param dist: distance (on mean) threshold for neighbors in PRM
        :return:
        '''
        self.max_actions = 1  # '''used to construct T'''
        if nodes:
            prod_nodes = itertools.product(nodes,self.nodes)
        else:
            prod_nodes = itertools.combinations(self.nodes,2)
        #to_nodes = self.nodes

        for node_i,node_j in prod_nodes:
            if node_i is node_j:
                continue
            dist_nodes = self.state_space.distance_mean(node_i, node_j)
            if dist_nodes > dist:
                continue
                # check whether the kripke condition is satisfied
            output = set(self.get_outputs(node_i, node_j))
            output_ends = set(self.get_outputs(node_i))
            output_ends |= set(self.get_outputs(node_j))

            if not (output.issubset(output_ends)):
                continue # kripke not satisfied

            # by now all conditions are satisfied
            # add edges in two directions
            self.edges.setdefault(node_i, []).append(node_j)
            self.edges.setdefault(node_j, []).append(node_i)

            self.edge_controllers.setdefault(node_i, []).append(
                Edge_Controller(self.motion_model, self.obs_model, node_i, node_j, self.Wx,
                                self.Wu, self.state_space))
            self.edge_controllers.setdefault(node_j, []).append(
                Edge_Controller(self.motion_model, self.obs_model, node_j, node_i, self.Wx,
                                self.Wu, self.state_space))

            if len(self.edges[node_i]) > self.max_actions:
                self.max_actions = len(self.edges[node_i])

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

    def compute_output_prob(self):
        '''Compute the probability over output set for each edge in the FIRM graph
        :return:
        '''
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

    def get_output_traj(self, traj):
        '''Returns a set of outputs generated by the trajectory.
            For now it returns only one output with the lowest position in the dictionary,
            even if multiple outputs are generated
            :param traj: list of belief_state(s)
            :return: key of regs
        '''
        output = 'null'
        for belief in traj:
            for (name, info) in self.regs.iteritems():
                poly = info[0]
                if poly.contains(belief.mean):
                    if self.regs.keys().index(output) > self.regs.keys().index(name):
                        output = name
        return output

    def plot_traj(self, traj, color):
        '''
        Plot a trajectory in belief space
        :param traj: traj = list of belief_state(s)
        :param color: color for argument of plot
        :return:
        '''
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


    def plot(self, ax):
        '''
        Plot the FIRM graph
        :param ax:  handle to plot
        :return:
        '''
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


    def abstract(self):
        '''Construct T by treating index of neigh as action number
        TODO: Update this for Hybrid-PBVI MDP definition
        TODO: (Sofie) => this is not used is it?
        # [R] MDP = abstraction'''
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
    ''' Consists of SLGR and SKF
    belief_space, motion_model, obs_model: Refer to classes in models.py
    # Wx = quadratic cost matrix of state used in LQR
    # Wu = quadratic cost matrix of input used in LQR'''
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


    def simulate_trajectory(self, b0):
        ''' Simulates trajectory starting from node_i to node_j
        :param b0: initial belief
        '''

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
    """
    Cross product between PRM (SPath) and DFA
    :param
    """
    def __init__(self,SPaths_object, formula, env, b_dist='U', n=50):
        """
        Initialize cross product of specification (DFA) with PRM graph
        :param SPaths_object:
        :param formula: Either Dict with DFA or scLTL spec
        :param env:
        :param b_dist:  'U' = uniform distribution with 'n' samples at every node
                        'PC' = use parent child info to generate belief nodes
        :param n: max number of belief points in a node.
        """
        if isinstance(formula, basestring):
            _dfa, self.dfsa_init, self.dfsa_final, self.proplist = formula_to_mdp(formula)
            self.fsa = Fsa()
            self.fsa.from_formula(formula)
            self.fsa.add_trap_state()
        elif isinstance(formula, dict):
            self.fsa = formula['fsa']
            self.dfsa_init = formula['init']
            self.dfsa_final = formula['final']
            self.proplist = formula['prop']
        else: raise TypeError

        # initialize with DFA and SPath object
        self.firm = SPaths_object
        self.env = env

        if b_dist == 'U':
            # initialize probability points
            probs = [0, 0.2, 0.5, 0.8, 1]  # what is this?
            self.probs_list = [probs for i in range(env.n_unknown_regs)]
            b_set = [i for i in product(*self.probs_list)]
            b_set = random.sample(b_set, n)
            self.b_reg_set = [env.get_reg_belief(list(i)) for i in b_set]
            self.b_prod_set = [env.get_product_belief(list(i)) for i in b_set]
            # True state of the regs used to simulate trajectories after policy is generated
            # x_e_true

            self.b_reg_set += [env.get_reg_belief(self.env.b_reg_init.tolist())]
            self.b_prod_set += [env.get_product_belief(self.env.b_reg_init.tolist())] # add initial
        else:
            assert False #not implemented TODO

        # accuracy requirement for convergence
        self.epsilon = 10**-5

        self.val = dict()
        self.active = dict()

        # values will point to values of _*_label_def below
        self.init = [(state,0) for (state, key) in self.fsa.init.items() if key == 1]

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

    def add_firm_node(self,n_nodes, dist_edge, means=list()):
        # add node to PRM/FIrM
        new_nodes = self.firm.new_nodes(n_nodes, means=means, append=True)
        # add edges in PRM/FIRM
        self.firm.make_edges(dist_edge, nodes=new_nodes)


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
        # Add only nodes that can actually be reached
        unvisited = [] # we use this list to keep track of nodes whose outgoing edges have not been included yet
        for (i_q,v) in self.init:
            # add all initial dfa states and initial graph stats (v=0)
            self.add_node((i_q, v))
            unvisited += [(i_q, v)]

        # add a virtual final node (-1,-1) can be used for breath first searches
            super(spec_Spaths, self).add_node((-1,-1))
            self.active[(-1,-1)] = False
            print('Added virtual final node = {v}'.format(v=(-1,-1)))


        while len(unvisited)>0:
            (i_q,i_v) = unvisited.pop(0)
            # TODO Add edges start from initial node
            for v_next in self.firm.edges[i_v]:
                # TODO complete this with obs and region labels
                # compute output of that vertex
                list_labels = self.firm.get_outputs(self.firm.nodes[v_next])

                # if null is included in the current labels then remove it
                try:
                    list_labels.remove('null')
                except ValueError: pass

                try:
                    bit_prop = self.fsa.bitmap_of_props((self.env.get_prop(list_labels[0]),)) # assume only one label at a time

                except IndexError:
                    bit_prop = 0



                for (orig_q_, q_next, label_dict) in self.fsa.g.out_edges(i_q, data=True):
                    if 0 in label_dict['input']:
                        unvisited = self._make_edge_to((i_q,i_v),(q_next, v_next), 'null', unvisited)
                        if bit_prop == 0:
                            continue # then we are done
                    if bit_prop in label_dict['input']:
                        # todo allow for more labels as input ('sample ^ obstacle') ==> sum them
                        unvisited = self._make_edge_to((i_q,i_v),(q_next, v_next), self.env.get_prop(list_labels[0]), unvisited)
        nodes = bfs(self,(-1,-1))
        u_nodes = OrderedDict() # this dictionary will give the sequence of nodes to iterate over
        for u in nodes:
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
            if input in dict_input  and next_n[1] == v:
                return next_n[0]

        raise ValueError

    def full_back_up(self, opts_old = None):
        t0 = time.clock()
        print('Do full back-up')

        for n in self.sequence:
            # do back up
            self.back_up(n[0],n[1],opts_old=opts_old)
        t1 = time.clock()
        print(t1-t0)
        return any(self.sequence.values())
        # boolean value giving whether or not the backups have converged

    def back_up(self, i_q, i_v, b=None, opts_old = None):
        '''
        This give the backup over all possible actions at
        a given point b if b is given or for all b points associated to (q,v)
        :param i_q: node in DFA
        :param i_v: node in firm
        :param b: belief point can either be a vector or None
        :return: updates internally the value function based on the expected next value functions
        '''

        if self.active[(i_q, i_v)] == False or self.sequence[(i_q, i_v)] == False:
            if self.sequence[(i_q, i_v)]:
                self.sequence[(i_q, i_v)] = False
            return

        # check first whether a backup is really needed
        # if all neighbors in the self graph have become inactive and
        # if it is false in the sequence then also this node can be set to false

        if isinstance(b,np.ndarray): # if no b is given then do it for all of them
            return self._back_up(i_q, i_v, b)
        else:
            alph_list = []
            best_edges = []
            opts = []
            # Get belief point from value function
            for b in self.val[(i_q, i_v)].b_prod_points:

                alpha_new, best_e, opt = self._back_up(i_q,i_v, b)
                alph_list += [alpha_new]
                best_edges += [best_e]
                opts += [opt]


            if isinstance(opts_old,dict):
                diff_opt = [abs(j - i) for i, j in zip(opts, opts_old.get((i_q, i_v),[-1]*len(opts)))]
                self.sequence[(i_q, i_v)] = sum(diff_opt) > (self.epsilon*len(self.val[(i_q, i_v)].b_prod_points))
            else :
                opts_old =dict()

            if self.sequence[(i_q, i_v)]: # you did change something, all neighbors could be affected

                for n_out in self.successors((i_q, i_v)):
                    if n_out in self.sequence.keys():
                        self.sequence[n_out] = True
                #print((i_q, i_v))
                opts_old[(i_q, i_v)] = opts
                alpha_mat = np.matrix(np.unique(np.array(np.concatenate(alph_list, axis=1)), axis=1))
                self.val[(i_q, i_v)].alpha_mat = alpha_mat
                self.val[(i_q, i_v)].best_edge = np.unique(best_edges)
            else:
                print('converged',(i_q, i_v) )
            return

    def _back_up(self, i_q, i_v, b):

        epsilon = self.epsilon
        # Set max alpha and best edge to current max/best (need this to avoid invariant policies)
        # Find index of best alpha from gamma set
        #index_alpha_init = np.argmax(self.val[(i_q, i_v)].alpha_mat.T * b)
        # Save best alpha vector
        opt = 0
        max_alpha_b_e = np.full_like(b, 0) #self.val[(i_q, i_v)].alpha_mat[:, index_alpha_init]
        # Save edge corresponding to best alpha vector
        best_e = None #self.val[(i_q, i_v)].best_edge[index_alpha_init]
        nf = (i_q, i_v) # source node
        # Foreach edge action
        for v_e in self.firm.edges[i_v]:
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
            if opt < (alpha_new.T * np.matrix(b)).item(0):
                max_alpha_b_e = alpha_new
                best_e = v_e
                opt = (alpha_new.T * np.matrix(b)).item(0)
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
                sum_o = sum_o + p_reach_goal_node* gamma_o_v[:, index]
            # Check if new alpha has a greater value
            if (opt + epsilon) < (sum_o.T * np.matrix(b)).item(0):
                # Update the max_alpha and best_edge
                max_alpha_b_e = sum_o
                best_e = -1 * (self.env.regs.keys().index(key) + 1)  # region 0 will map to edge -1
                opt = (sum_o.T * np.matrix(b)).item(0)

        # Sanity check that alpha <= 1
        # if not (max_alpha_b_e <= 1).all():
        #     print('warning ' , max_alpha_b_e, best_e)
        #     assert False

        return (max_alpha_b_e, best_e, opt)

    def plot_node(self,node,region, b=None,ax = None):
        '''
        Give a one D plot of the value function at a given node by varying the uncertainty of the given region.
        :param node: node (q=node of DFA ,v= node of FIRM) in the
        :param region: The region key for example 'R4'.
        :param b: list of probabilities associated to each of the regions.
        If none take the initial probabilities.
        :param ax: ahndle for axes on which the plot can be given,
         if none automatically generate the plot. Otherwise add plot to the handle.
        :return: Returns a plot
        '''

        if b is None:
            # no b is given.
            self.env.b_reg_init = np.matrix(self.b_reg_init).T
            print(self.b_reg_init)


def bfs(graph, start):
    visited, queue = [], [start]
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited += [vertex]
            add_to = list((set(graph.predecessors(vertex)) - set(visited))- set(queue))
            queue += add_to

    print('visited', visited)
    return visited


#prod_.init = [('0',0,)]
from copy import copy
def plot_results(prod,ax):
    '''
    Give a plot of the optimizers of the current graph.
    Based on the firm graph, but nodes now include also info on the state of the DFA
    :param prod: product of DFA and FIRM
    :type prod: spec_Spaths
    :return:
    '''
    # TODO TODO FiniSH WHERE I STOPPED CODING >>> make sure to only plot directed edges for possible optimal actions.
    # build the nodes from the once that are actually reachable:
    # start from the initial node
    nodes = dict()  # empty dict with nodes as keys and values: (x,y),
    obs = dict()   #  empty dict with nodes as keys, values: set of obs actions
    edges = set()  # empty set with elements (x1,y1,x2,y2) with node from   (x1,y1)  and node to (x2,y2)
     # list of nodes for which edges needs to be accounted for, initialized with
                        # a list of initial node tuples (i_q, i_v)
    unvisited = copy(prod.init)
    visited = []
    while unvisited:
        n = unvisited.pop(0)
        visited += [n]
        (i_q, i_v) = n
        opt = np.unique(prod.val[(i_q, i_v)].best_edge)
        # split into obs actions and transitions actions
        obs_actions  = filter(lambda i: i<0, opt) # decide to observe a neigborhood
        tr_actions  = filter(lambda i: i>=0, opt) # decide to transition to a new node
        # find (x,y)
        x = np.ravel(prod.firm.nodes[i_v].mean)[0]
        y = np.ravel(prod.firm.nodes[i_v].mean)[1]
        nodes[i_v] = (x,y)
        obs[i_v] = obs.get(i_v,set())|set(obs_actions)
        edges |= {(i_v,i_next) for i_next in tr_actions}
        # to find the nodes (composed of i_q,i_v) that have not yet been added and
        # that are accessible from this node based ont he transition actions, check transitions in prod

        for n_next in prod[n]:
            if n_next[1] in tr_actions:
                if (not n_next in unvisited) and (not n_next in visited):
                    unvisited.extend([(n_next)])
        for i_v in nodes:
            if i_v < 10:
                plt.text(nodes[i_v][0] - 0.04, nodes[i_v][1] - 0.05, str(i_v),
                         color='black', backgroundcolor='grey')
            else:
                plt.text(nodes[i_v][0] - 0.09, nodes[i_v][1] - 0.05, str(i_v),
                         color='black', backgroundcolor='grey')
    for (start,dest) in edges:
            plt.plot([nodes[start][0],nodes[dest][0]],[nodes[start][1],nodes[dest][1]],color='black')

            plt.arrow(nodes[start][0],nodes[start][1],.7*(nodes[dest][0]-nodes[start][0]),.7*(nodes[dest][1]-nodes[start][1]), head_width=0.2, head_length=.2, fc='k', ec='k')



    print('Nodes that can be pruned: {n}'.format(n=sorted(filter(lambda i: not i[1] in nodes, prod.nodes))))
    ax.set_xlim(prod.firm.state_space.x_low[0], prod.firm.state_space.x_up[0])
    ax.set_ylim(prod.firm.state_space.x_low[1], prod.firm.state_space.x_up[1])
    print('number of active nodes = {n}, percent of orig. nodes = {o} %'.format(n=len(nodes),o=100*len(nodes.keys())/len(prod.nodes)))
    print('number of active edges = {n}, percent of orig. edges = {o} %'.format(n=len(edges),o=100*len(edges)/len(prod.edges),))
