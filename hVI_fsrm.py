"""
FSRM = Feedback State RoadMap
This file gives the class for the generation of a sampling based path planner
that can be used together with the HVI tools.
"""
import itertools
import itertools as it
import logging
import random
import warnings
from collections import *
from copy import copy
from itertools import product
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.spatial import distance

import aux as rf
from fsa import Fsa
from fsa import formula_to_mdp
from hVI_models import Det_SI_Model, State_Space, State
from hVI_types import Gamma

logger = logging.getLogger(__name__)


class SPaths(object):
    """
    Roadmap class
    """

    def __init__(self, state_space, motion_model, Wx, Wu, regs, output_color, ax, nearest_n=0):
        """
        Construct Roadmap object with

        :param state_space: Refer to classes in models.py
        :type state_space: State_Space
        :param motion_model: Refer to classes in models.py
        :type motion_model: Det_SI_Model
        :param Wx: quadratic cost matrix of state used in LQR
        :type Wx: np.array
        :param Wu: quadratic cost matrix of input used in LQR
        :type Wu: np.array
        :param regs: same as RSS definition
        :param output_color: e.g., output_color =\n {-1:'black', 0:'blue', 1:'green', 2:'red'}
        :param ax: handle for plotting stuff, can be generated as: fig = plt.figure(0); ax = fig.add_subplot(111, aspect='equal')
        :param nearest_n: default = 0, n>0 implies that nodes will be connected to the n nearest neigbors
        """

        self.nearest_n = nearest_n
        # self.max_actions = len(self.edges[node_i])
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
        self.node_controllers = dict()
        # TODO the use of ordered dicts here is potentially slowing everything down.
        #
        self.edges = OrderedDict()  # key=node_id and value=list of node_ids of neighbors
        self.edge_controllers = OrderedDict()
        self.edge_output_prob = OrderedDict()
        self.T_list = None
        self.Tz_list = None
        self.n_particles = 1

    def make_nodes_edges(self, number, edges_dist, init=None, means=list(),**kwargs):
        """
        Initiate the nodes and edges in the roadmap

        :param means:
        :param number: number of nodes
        :param edges_dist: max distance of connected nodes
        :param init: initial node, i.e., starting position of the robot.
        :return:
        """
        self.sample_nodes(number, init=init, means=means, **kwargs)

        self.make_edges(edges_dist)


    def sample_nodes(self, n_nodes, means=list(), append=False, init=None, **kwargs):
        """
        Sample nodes in belief space and also generate node_controllers

         :param n_nodes  = number of nodes to sample in graph
         :param means = list of means for placing nodes in the roadmap
         :param append = False erases all previous nodes whereas True adds more nodes to existing graph
         :param init = initial node for the road map
         :keyword min_dist = Samples within min_dist range will be rejected unless they produce non-NULL outputs
                   can be set to 0.0 to disable pruning
         :keyword sample : grid => sample over grid
        """
        if kwargs.get('sample',' ') == 'grid':
            grid = True
        else:
            grid = False

        if len(means) > 0 and n_nodes < len(self.regs):
            if append is False:
                raise ValueError('Number of samples cannot be less than n_regs')

        # make a list of nodes
        nodes = []
        if append is False:
            self.nodes = []  # clear previous nodes/edges
            self.edges = OrderedDict()
            self.node_controllers = dict()
            self.edge_controllers = OrderedDict()

        if len(means) > 0:  # => if you already have a precomputed node to add use that one first.
            pass
        elif not append:  # first use init
            if isinstance(init, np.ndarray):
                nodes += [State(init)]  # add first mean for graph node
            for reg in self.regs:
                if (self.regs[reg][2] is not 'obs') or (self.regs[reg][1] < 1):
                    nodes += [self.state_space.sample_new_state_in_reg(self.regs[reg][0])]

        while len(means) > 0:
            nodes += [self.state_space.new_state(means.pop(0))]

        added_nodes = []
        while n_nodes > 0:
            if nodes:
                node = nodes.pop(0)  # keep taking first node until empty
            else:
                # Implemented rejection sampling to avoid nodes in obs => why would you do that?
                # you want to avoid samples in regions that we know to be obstacles,
                # but if they could be obstacles and it is not sure, then it makes more sense to keep samples in them
                if grid:
                    node = self.state_space.sample_new_state_from_grid(delta=0.4)

                else:

                    node = self.state_space.sample_new_state()
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

            assert 'min_dist' not in kwargs
            # not implemented
            # TODO: implement something like this, Rohans code seems to have dissapeared here.

            # Set Co-variance
            self.nodes.append(node)
            self.node_controllers[node] = Node_Controller(self.motion_model, self.obs_model,
                                                          node, self.Wx, self.Wu,
                                                          self.state_space)

            added_nodes += [node]
            n_nodes += -1
        return added_nodes

    def prune_nodes(self, keeplist):
        # todo: pruning over new class structure
        # due to the current structure of the nodes and edges actually pruning FIRM is computationally intensive.
        #  Therefore we have not implemented it. To enable a good implementation,
        #  the SPaths object should inherent nodes and edges and its handles from the networkx packages.
        # right now , nodes will be removed or pruned from the product structure only.

        rem_nodes = [n for n in self.nodes if n not in keeplist]
        for  n in rem_nodes:
            self.nodes.remove(n)

        # remove transitions/edges from and to nodes, that are no longer in the set of nodes.
        for from_node in self.edges.keys():
            if from_node in self.nodes:
                self.edges[from_node] = [tonode for tonode in self.edges[from_node] if tonode in self.nodes]
            else:
                self.edges.pop(from_node)

    def make_edges(self,dist,  nodes=list(), give_connected=False ):
        """
        Construct edges for self.nodes within distance dist and generate edge controllers

        :param dist: distance (on mean) threshold for neighbors in PRM
        :param nodes: list of nodes from which edges are still missing.
        Default is none, in which case all nodes have missing edges.
        :param give_connected: Give the parent nodes newly connected to the nodes
        :return:
        """
        unvisited = []  # these are nodes whose outgoing edges need to be checked again in the specification roadmap.
        # Therefore they are unvisited for the product creation.

        self.max_actions = 5

        if self.nearest_n<=0:
            if nodes:
                prod_nodes = itertools.product(nodes, self.nodes)

            else:
                if give_connected:
                    raise ValueError  # can only be true when nodes is not empty
                prod_nodes = itertools.combinations(self.nodes, 2)
            # to_nodes = self.nodes

            for node_i, node_j in prod_nodes:
                if node_i is node_j:
                    continue
                if len(self.edges.get(node_i,list())) >= self.max_actions and len(self.edges.get(node_j,list())) >= self.max_actions:
                    continue

                dist_nodes = self.state_space.distance_mean(node_i, node_j)
                if dist_nodes > dist:
                    continue

                if not self._connect(node_i, node_j, dist_nodes):
                    continue

                if give_connected:
                    unvisited += [node_j]




        elif self.nearest_n>0:

            all_nodes = np.concatenate([node.mean for node in self.nodes], axis=1)
            l_nodes = len(self.nodes)
            if nodes:
                new_nodes = np.concatenate([node.mean for node in nodes], axis=1)

            else:
                new_nodes = all_nodes
                nodes = self.nodes
                if give_connected:
                    raise ValueError  # can only be true when nodes is not empty
                prod_nodes = itertools.combinations(self.nodes, 2)

            # to_nodes = self.nodes

            dist_mat = distance.cdist(new_nodes.T, all_nodes.T)

            for i, node in enumerate(nodes):
                closest = sorted(range(l_nodes), key=lambda k: dist_mat[i,k])
                # sort based on distance to i-th node
                neigh = 0
                while len(closest) > 0 and neigh < self.nearest_n:
                    j = closest.pop(0)

                    if not self._connect(node, self.nodes[j], dist_mat[i,j]):
                        continue

                    if give_connected:
                        unvisited += [self.nodes[j]]
                    neigh += 1



        else:
            raise NotImplementedError

        if give_connected:
            return unvisited

    def _connect(self, node_i, node_j,dist_nodes):
        if node_i == node_j:
            return False
        # check whether the Kripke condition is satisfied
        output = set(self.get_outputs(node_i, node_j))
        output_ends = set(self.get_outputs(node_i))
        output_ends |= set(self.get_outputs(node_j))

        if not (output.issubset(output_ends)):
            return False  # kripke not satisfied

        # by now all conditions are satisfied
        # add edges in two directions
        self.edges.setdefault(node_i, []).append(node_j)
        self.edges.setdefault(node_j, []).append(node_i)

        self.edge_controllers[(node_i, node_j)] = Edge_Controller(self.motion_model, self.obs_model,
                                                                  node_i, node_j,
                                                                  self.Wx,
                                                                  self.Wu, self.state_space, dist=dist_nodes)
        self.edge_controllers[(node_j, node_i)] = Edge_Controller(self.motion_model, self.obs_model,
                                                                  node_j, node_i,
                                                                  self.Wx,
                                                                  self.Wu, self.state_space, dist=dist_nodes)
        return True


    def intersect(self, box, src, dest):
        """
        Check whether there is an intersection with a region

        :param box: region box
        :param src: source state
        :param dest: destination state
        :return:  Boolean value
        """
        diff = dest - src
        ranges = np.append(*box.bounding_box, axis=1)

        low, high = box.bounding_box

        u = np.zeros((2,))
        v = np.ones((2,))

        if abs(diff[0]) < np.finfo(float).eps:  # constant along the x-axis
            if not (ranges[0, 0] <= src[0] <= ranges[0, 1]):
                return False
        else:
            u[0] = max(min((low[0] - src[0]) / diff[0], (high[0] - src[0]) / diff[0]), 0)
            v[0] = min(max((low[0] - src[0]) / diff[0], (high[0] - src[0]) / diff[0]), 1)

        if abs(diff[1]) < np.finfo(float).eps:  # constant along the y-axis
            if not (ranges[1, 0] <= src[1] <= ranges[1, 1]):
                return False
        else:
            u[1] = max(min((low[1] - src[1]) / diff[1], (high[1] - src[1]) / diff[1]), 0)
            v[1] = min(max((low[1] - src[1]) / diff[1], (high[1] - src[1]) / diff[1]), 1)
        assert (v <= 1).all()
        assert (u >= 0).all()
        return np.max(u) <= np.min(v)

    def compute_output_prob(self):
        """Compute the probability over output set for each edge in the FIRM graph
        For the deterministic model, this will assign Boolean truth values to potential labellings
        TODO: relate this to our choice of Kripke structure
        :return:
        """
        raise NotImplementedError
        # this error is raised since the structure of the edge controllers has been changed.
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
        """
        Returns a set of outputs generated by the trajectory.
            For now it returns only one output with the lowest position in the dictionary,
            even if multiple outputs are generated

            :param traj: list of belief_state(s)
            :return:
        """
        output = 'null'
        for belief in traj:
            for (name, info) in self.regs.iteritems():
                poly = info[0]
                if poly.contains(belief.mean):
                    if self.regs.keys().index(output) > self.regs.keys().index(name):
                        output = name
        return output

    def plot_traj(self, traj, color):
        """
        Plot a trajectory in belief space

        :param traj: traj = list of belief_state(s)
        :param color: color for argument of plot
        :return:
        """
        for i in range(len(traj) - 1):
            if isinstance(traj[i], State):
                try:
                    x = [np.ravel(traj[i].mean)[0], np.ravel(traj[i + 1].mean)[0]]
                    y = [np.ravel(traj[i].mean)[1], np.ravel(traj[i + 1].mean)[1]]
                except:
                    x = [np.ravel(traj[i].mean)[0], np.ravel(traj[i + 1])[0]]
                    y = [np.ravel(traj[i].mean)[1], np.ravel(traj[i + 1])[1]]

            else:
                x = [np.ravel(traj[i])[0], np.ravel(traj[i + 1])[0]]
                y = [np.ravel(traj[i])[1], np.ravel(traj[i + 1])[1]]
            # if color == 'white':
            #     color = 'black'
            return self.ax.plot(x, y, color, ms=20, linewidth=3.0)

    def plot(self, ax):
        """
        Plot the FIRM graph

        :param ax:  handle to plot
        :return:
        """
        for node_i in self.nodes:
            try:
                neigh = self.edges[node_i]
            except KeyError:
                print("error in self.edges")
                continue
            for node_j in neigh:
                x = [np.ravel(node_i.mean)[0], np.ravel(node_j.mean)[0]]
                y = [np.ravel(node_i.mean)[1], np.ravel(node_j.mean)[1]]
                ax.plot(x, y, 'b')

        scale = 3
        rf.plot_nodes(self.nodes)


        for i in range(len(self.nodes)):
            # including the code in the following lines in the previous for loop doesnt work
            if i < 10:
                plt.text(np.ravel(self.nodes[i].mean)[0] - 0.04, np.ravel(self.nodes[i].mean)[1] - 0.05, str(i),
                         color='black', backgroundcolor='grey')
            else:
                plt.text(np.ravel(self.nodes[i].mean)[0] - 0.09, np.ravel(self.nodes[i].mean)[1] - 0.05, str(i),
                         color='black', backgroundcolor='grey')
        ax.set_xlim(self.state_space.x_low[0], self.state_space.x_up[0])
        ax.set_ylim(self.state_space.x_low[1], self.state_space.x_up[1])
        for (name, info) in self.regs.iteritems():
            hatch = False
            fill = True
            if name is not 'null':
                rf.plot_region(ax, info[0], name, info[1], self.output_color[name], hatch=hatch, fill=fill)
        # plt.show()


class Node_Controller(object):
    """
    Basic node controller for the deterministic model
    """

    def __init__(self, motion_model, obs_model, node, Wx, Wu, state_space):
        """
        Initialize node controller

        :param motion_model: Deterministic motion model
        :type motion_model: Det_SI_Model
        :param obs_model: This is  None for the deterministic models, see  best/hVI_fsrm.py:56
        :param node: Node in the roadmap at which this controller is implemented
        :param Wx: The weight factor for LQR for the state
        :param Wu: The weight matrix for the input
        :param state_space: The statespace over which it is computed
        :type state_space: State_Space
        """
        self.motion_model = motion_model
        self.obs_model = obs_model  # = None
        self.A = self.motion_model.getA(node)
        self.B = self.motion_model.getB(node)
        self.node = node
        self.Wx = Wx
        self.Wu = Wu
        self.state_space = state_space
        # set control gain
        # S = np.mat(dare(self.A, self.B, self.Wx, self.Wu))
        self.motion_model.getLS(Wx, Wu)
        self.Ls = self.motion_model.Ls  # (self.B.T * S * self.B + self.Wu).I * self.B.T * S * self.A

    def simulate_trajectory(self, b0):
        """
        Simulates trajectory form node_i to node_j
        :param b0: initial belief state
        """
        traj = [b0]
        while not self.state_space.distance(traj[-1], self.node) < 0.1:
            # -->> Get control, apply control, get observation and apply observation
            b = traj[-1]
            # Get control
            u_k = -self.Ls * (np.mat(b) - self.node.mean)
            # Apply control/predict (Esq. 75-76, 98-99)
            # x^-_k+1 = A * x^+_k + B * u; P^-_k+1 = A * P^+_k *A^T + G Q G^T
            bnew_pred = self.motion_model.evolve(b, u_k)
            traj.append(self.state_space.new_state(bnew_pred))

        return traj


class Edge_Controller(object):
    """
    Time varying LQG controller

    """

    def __init__(self, motion_model, obs_model, node_i, node_j, Wx, Wu, state_space, dist=1):
        """
        Initialize the edge control

        :param motion_model:  Refer to classes in models.py
        :type motion_model: Det_SI_Model
        :param obs_model:  Refer to classes in models.py
        :param node_i: source node
        :param node_j: destination node
        :param Wx: quadratic cost matrix of state used in LQR ==> unused
        :param Wu: quadratic cost matrix of input used in LQR ==> unused
        :param state_space:
        """
        self.motion_model = motion_model
        self.obs_model = obs_model  # = None
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
            # A, B, S_next = map(np.mat, [self.motion_model.getA(self.traj_d[t]),
            #                            self.motion_model.getB(self.traj_d[t]),
            #                            S[t + 1, :, :]])
            # L = (B.T * S_next * B + self.Wu).I * B.T * S_next * A

            self.L[t, :, :] = self.motion_model.Ls
            # S[t, :, :] = self.Wx + A.T * S_next * (A - B * L)
            if t == 0:
                break

        self.prob = 0.99 ** dist

    def simulate_trajectory(self, b0):
        """ Simulates trajectory starting from node_i to node_j
        NOT IMPLEMENTED

        :param b0: initial belief
        """
        # traj = [self.node_i]
        traj = [b0]
        for t in range(self.N - 1):
            b = traj[-1]

            u = -self.L[t, :, :] * (b.mean - self.traj_d[t]) + self.u0[t]
            bnew_pred = self.motion_model.evolve(b, u)
            # Update Belief
            traj.append(self.state_space.new_state(bnew_pred))
        return traj


class Spec_Spaths(nx.MultiDiGraph):
    """
    Cross product between PRM (SPath) and DFA, that is used to compute the value iterations over.
    """

    def __init__(self, SPaths_object, formula, env, b_dist='U', n=50):
        """
        Initialize cross product of specification (DFA) with PRM graph
        :param SPaths_object:
        :param formula: Either Dict with DFA or scLTL spec
        :param env:
        :param b_dist:  'U' = uniform distribution with 'n' samples at every node
                        'PC' = use parent child info to generate belief nodes (NOT IMPLEMENTED)
        :param n: max number of belief points in a node.
        """
        if isinstance(formula, basestring):
            self.dfsa_init, self.dfsa_final, self.proplist = formula_to_mdp(formula)
            self.fsa = Fsa()
            self.fsa.from_formula(formula)
            self.fsa.add_trap_state()
        elif isinstance(formula, dict):
            self.fsa = formula['fsa']
            self.dfsa_init = formula['init']
            self.dfsa_final = formula['final']
            self.proplist = formula['prop']
        else:
            raise TypeError

        # initialize with DFA and SPath object
        self.firm = SPaths_object
        self.env = env

        # initialize the belief points
        if b_dist == 'U':
            # initialize probability points

            probs = [0, 0.2, 0.5, 0.8, 1]  # for the individual dimensions
            #  probs is a predefined set of belief points
            # TODO: make this an argument of the function
            # that can be used as a default
            self.probs_list = [probs for i in range(env.n_unknown_regs)]
            #  over all dimensions
            b_set = [i for i in product(*self.probs_list)]
            b_set = random.sample(b_set, n)
            self.b_reg_set = [env.get_reg_belief(list(i)) for i in b_set]
            self.b_prod_set = [env.get_product_belief(list(i)) for i in b_set]
            # True state of the regs used to simulate trajectories after policy is generated
            # x_e_true

            self.b_reg_set += [env.get_reg_belief(self.env.b_reg_init.tolist())]
            self.b_prod_set += [env.get_product_belief(self.env.b_reg_init.tolist())]  # add initial

        elif b_dist == 'PC':
            # compute belief points via forward reachable set,
            # i.e., parent child info
            assert False

        elif b_dist == 'R':
            per_dim = math.ceil(n**(1.0/env.n_unknown_regs))
            # initialize probability points
            probs = np.linspace(0,1,per_dim)  # for the individual dimensions
            #  probs is a predefined set of belief points
            # that can be used as a default
            self.probs_list = [probs for i in range(env.n_unknown_regs)]
            #  over all dimensions
            b_set = [i for i in product(*self.probs_list)]

            b_set = random.sample(b_set, n)
            self.b_reg_set = [env.get_reg_belief(list(i)) for i in b_set]
            self.b_prod_set = [env.get_product_belief(list(i)) for i in b_set]
            # True state of the regs used to simulate trajectories after policy is generated
            # x_e_true

            self.b_reg_set += [env.get_reg_belief(self.env.b_reg_init.tolist())]
            self.b_prod_set += [env.get_product_belief(self.env.b_reg_init.tolist())]  # add initial


        else:
            assert False  # not implemented TODO: extra feature

        # accuracy requirement for convergence
        self.epsilon = 10 ** -5

        self.val = dict()
        self.active = dict()

        # values will point to values of _*_label_def below
        firm_init = self.firm.nodes[0]
        self.init = [(state, firm_init) for (state, key) in self.fsa.init.items() if key == 1]

        types = [
            {'name': 'input',
             'values': {'null'} | set(self.fsa.props.keys()),
             'setter': True,
             'default': 'null'}]
        reg_names = [{'name': 'reg',
                      'values': {'null'} | set(self.env.regs.keys()),
                      'setter': True,
                      'default': 'null'}]

        super(Spec_Spaths, self).__init__(state_label_types=reg_names, edge_label_types=types)
        self.sequence = self.create_prod()

    def add_firm_node(self, n_nodes, dist_edge, means=list(), **kwargs):
        """
        Add a node to the specification roadmap directly.
            - Node will be added as a ROADMAP node and in this specification roadmap
        :param n_nodes: Number of new nodes to be added
        :param dist_edge: The minimal distance between two connected nodes
        :param means: an optional list of means from which the new road map nodes are selected
        :return:  Updated ROADMAP and SPEC-ROADMAP
        """
        # add node to PRM/FIRM
        new_nodes = self.firm.sample_nodes(n_nodes, means=means, append=True,**kwargs)
        # print("added nodes ", new_nodes)
        # add edges in PRM/FIRM
        unvisited = self.firm.make_edges(dist_edge, nodes=new_nodes, give_connected=True)
        self.sequence = self.create_prod(unvisited_v=unvisited)
        return new_nodes


    def _make_edge_to(self, from_pair, node_pair, label, unvisited):
        """
        Make edge in th SPEC-ROADMAP. Thisedge is created to enable faster backups.
        :param from_pair: source node in SPEC-ROADMAP
        :param node_pair: TARGET node in SPEC-ROADMAP
        :param label:
        :param unvisited: The list of unvisted nodes. This list is extended with nodes that
                            are childeren of the current nodes
        :return:
        """
        if node_pair not in self.nodes():
            self.add_node(node_pair)
            if node_pair[0] in self.fsa.final:
                self.val[node_pair].alpha_mat[:, 0] = 1
                self.active[node_pair] = False
                self.add_edge(node_pair, (-1, -1))
                # node (-1,-1) is the virtual node used for computations
            else:
                unvisited += [node_pair]

        self.add_edge(from_pair, node_pair, input=label)

        return unvisited

    def create_prod(self, unvisited_v=None):
        """
        Initialize the product  between the DFA for the specification (SPEC) and
        the ROADMAP for the robot planning the get the SPEC-ROADMAP.

        :return: return the SPEC-ROADMAP
        """

        if unvisited_v:
            print('update product')
            # Add only nodes that can actually be reached
            unvisited = []  # we use this list to keep track of nodes whose outgoing edges have not been included yet
            for (i_q, v) in itertools.product(self.fsa.g.nodes, unvisited_v):
                # add all initial dfa states and initial graph stats (v=0)
                if (i_q, v) in self.nodes:
                    unvisited += [(i_q, v)]

        else:

            # Add only nodes that can actually be reached
            unvisited = []  # we use this list to keep track of nodes whose outgoing edges have not been included yet
            for (i_q, v) in self.init:
                # add all initial dfa states and initial graph stats (v=0)
                self.add_node((i_q, v))
                unvisited += [(i_q, v)]

                # add a virtual final node (-1,-1) can be used for breath first searches
                super(Spec_Spaths, self).add_node((-1, -1))
                self.active[(-1, -1)] = False

        while len(unvisited) > 0:
            (i_q, i_v) = unvisited.pop(0)
            for v_next in self.firm.edges[i_v]:
                # compute output of that vertex
                list_labels = self.firm.get_outputs(v_next)
                try:
                    list_labels.remove('null')
                except ValueError:
                    pass

                try:
                    bit_prop = self.fsa.bitmap_of_props((self.env.get_prop(list_labels[0]),))
                    # assume only one label at a time

                except IndexError:
                    bit_prop = 0

                for (orig_q_, q_next, label_dict) in self.fsa.g.out_edges(i_q, data=True):
                    if 0 in label_dict['input']:
                        unvisited = self._make_edge_to((i_q, i_v), (q_next, v_next), 'null', unvisited)
                        if bit_prop == 0:
                            continue  # then we are done
                    if bit_prop in label_dict['input']:
                        # todo allow for more labels as input ('sample ^ obstacle') ==> sum them
                        unvisited = self._make_edge_to((i_q, i_v),
                                                       (q_next, v_next),
                                                       self.env.get_prop(list_labels[0]), unvisited)

        nodes = bfs(self, (-1, -1))
        u_nodes = OrderedDict()  # this dictionary will give the sequence of nodes to iterate over
        for u in nodes:
            u_nodes[u] = True

        u_nodes[(-1, -1)] = False
        for u in u_nodes:
            if not self.active[u]:
                u_nodes[u] = False
        return u_nodes

    def add_node(self, n, attr_dict=None, check=True, **attr):
        """
        add_node to SPEC-ROADMAP

        :param n: node to be added
        :param attr_dict: the dictionary belonging to it
        :param check: Check is needed for adding nodes in networkx
        :param attr: other additional attributes inhereted from networkx
        :return: updated graph
        """
        # add node n to graph
        list_labels = self.firm.get_outputs(n[1])
        if len(list_labels) > 1:
            list_labels.remove('null')

        super(Spec_Spaths, self).add_node(n, attr_dict=attr_dict, check=check, reg=list_labels[-1])
        self.val[n] = Gamma(self.b_prod_set, self.b_reg_set)  # added i_q,i_v
        self.active[n] = True
        # print("added ", n)

    def rem_node(self, n):
        """Delete a node

        :param n: node to be deleted
        """
        # (n=node)=> remove node from prod graph
        super(Spec_Spaths, self).remove_node(n)

        self.active.__delitem__(n)
        self.val.__delitem__(n)


    def prune(self, keep_list=None, rem_list=None):
        print('start')
        rem = []
        if keep_list:
            old_nodes = copy(self.nodes)
            for node in old_nodes:
                if node in keep_list or node == (-1, -1):
                    pass
                else:
                    if node in self.nodes:
                        rem += [node]

        elif rem_list:
            raise NotImplementedError
        else:
            raise ValueError

        for node in rem:
            self.rem_node(node)

        nodes = bfs(self, (-1, -1))
        prm_nodes = set(n[1] for n in nodes)
        self.firm.prune_nodes(list(prm_nodes))


        u_nodes = OrderedDict()  # this dictionary will give the sequence of nodes to iterate over
        for u in nodes:
            u_nodes[u] = True

        u_nodes[(-1, -1)] = False
        for u in u_nodes:
            if not self.active[u]:
                u_nodes[u] = False
        self.sequence = u_nodes

    def find_edge(self, n, input_let, v=None):
        """ Find an edge based on the input letters

        :param n: starting node (dfa_state, roadmap state)
        :param input_let: input letter
        :param v: the next roadmap state, Default=None
        :return I_q, the next DFA state if v is not None, \n
            Else if v is None the next SPEC-ROADMAP state is given as: (dfa_state, roadmap state)
        """
        if v is None:
            for (n_, next_n, dict_input) in self.out_edges({n}, data='input'):
                # print(dict_input,input_let,'ll')
                if dict_input == input_let:
                    return next_n

        for (n_, next_n, dict_input) in self.out_edges({n}, data='input'):

            if input_let in dict_input and next_n[1] == v:
                return next_n[0]

        print(v)

        print(n, input_let, self.out_edges({n}, data='input'))
        raise ValueError

    def full_back_up(self, opts_old=None):
        """
        This function does a full back-up over the backward reachable SPEC-ROADMAP

        :param opts_old: Default = None \n
        :return: Boolean value giving whether or not the backups have converged
        """
        for n in self.sequence:
            # do back up
            self.back_up(n[0], n[1], opts_old=opts_old)

        return any(self.sequence.values())

    def back_up(self, i_q, i_v, b=None, opts_old=None):
        """
        This give the backup over all possible actions at
        a given point b if b is given or for all b points associated to (q,v). updates internally the value function based on the expected next value functions

        :param i_q: node in DFA
        :param i_v: node in firm
        :param b: belief point can either be a vector or None
        :param opts_old: aux variable with previous computed values of the nodes.
         This is used to eliminate nodes that have converged from the sequence of to be backed-up nodes
        """

        if self.active.setdefault((i_q, i_v), True) == False \
                or self.sequence.setdefault((i_q, i_v), True) == False:
            if self.sequence[(i_q, i_v)]:
                self.sequence[(i_q, i_v)] = False
            return

        # check first whether a backup is really needed
        # if all neighbors in the self graph have become inactive and
        # if it is false in the sequence then also this node can be set to false

        if isinstance(b, np.ndarray):  # if no b is given then do it for all of them
            return self._back_up(i_q, i_v, b)
        else:
            alph_list = []
            best_edges = []
            opts = []
            # Get belief point from value function

            for b in self.val[(i_q, i_v)].b_prod_points:
                alpha_new, best_e, opt = self._back_up(i_q, i_v, b)
                alph_list += [alpha_new]
                best_edges += [best_e]
                opts += [opt]

            if isinstance(opts_old, dict):
                diff_opt = [abs(j - i) for i, j in zip(opts, opts_old.get((i_q, i_v), [-1] * len(opts)))]
                self.sequence[(i_q, i_v)] = sum(diff_opt) > (self.epsilon * len(self.val[(i_q, i_v)].b_prod_points))
            else:
                opts_old = dict()

            if self.sequence[(i_q, i_v)]:  # you did change something, all neighbors could be affected

                for n_out in self.successors((i_q, i_v)):
                    if n_out in self.sequence.keys():
                        self.sequence[n_out] = True
                opts_old[(i_q, i_v)] = opts

                self.val[(i_q, i_v)].alpha_mat = np.concatenate(alph_list, axis=1)
                self.val[(i_q, i_v)].prune()
                self.val[(i_q, i_v)].best_edge = np.unique(best_edges)

    def _back_up(self, i_q, i_v, b):
        """
        Back up operation for a given node.

        :param i_q: Discrete node in DFA associated to the node
        :param i_v: Discrete node in roadmap associated to the node
        :param b: list of belief points
        :return: best value function (max_alpha_b_e, best_e, opt)
        """
        epsilon = self.epsilon
        # Set max alpha and best edge to current max/best (need this to avoid invariant policies)
        # Find index of best alpha from gamma set
        # index_alpha_init = np.argmax(self.val[(i_q, i_v)].alpha_mat.T * b)
        # Save best alpha vector
        opt = 0
        max_alpha_b_e = np.full_like(b, 0)  # self.val[(i_q, i_v)].alpha_mat[:, index_alpha_init]
        # Save edge corresponding to best alpha vector
        best_e = None  # self.val[(i_q, i_v)].best_edge[index_alpha_init]
        nf = (i_q, i_v)  # source node
        # Foreach edge action
        for v_e in self.firm.edges[i_v]:
            # Get probability of reaching goal vertex corresponding to current edge
            # p_reach_goal_node = firm.reach_goal_node_prob[i_v][i_e]
            p_reach_goal_node = self.firm.edge_controllers[
                (i_v, v_e)].prob  # = 0.99  # TODO Get this from Petter's Barrier Certificate
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
                        q_z_o = self.find_edge(nf, self.env._regs[z][2], v=v_e)
                        # q_z_o = np.argmax(self.dfsa.T(self.proplist[self.env._regs[z][2]])[i_q, :])
                    else:
                        # new q = current q
                        q_z_o = self.find_edge(nf, 'null', v=v_e)
                    gamma_e = np.diag(np.ravel(O[i_o, :])) * np.matrix(self.val[(q_z_o, v_e)].alpha_mat)
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

        p_reach_goal_node = 0.99  # TODO: remove  hard coding of transition prob for observations
        # For each obs action (iterate through every region that we can observe)
        for key, info in self.env.regs.iteritems():
            # Get observation matrix as defined in the paper
            O = self.env.get_O_reg_prob(key, i_v.mean)
            # Initialize sum over observations to zero
            sum_o = np.zeros([2 ** self.env.n_unknown_regs, 1])
            # Iterate over possible observations/Labels (True, False)
            for i_o in range(2):
                # Get new Gamma set
                gamma_o_v = np.diag(np.ravel(O[i_o, :])) * np.matrix(self.val[(i_q, i_v)].alpha_mat)
                # Find index of best alpha in the new Gamma set
                index = np.argmax(gamma_o_v.T * b)
                # Add the best alpha to the summation
                sum_o = sum_o + p_reach_goal_node * gamma_o_v[:, index]
            # Check if new alpha has a greater value
            if (opt + epsilon) < (sum_o.T * np.matrix(b)).item(0):
                # Update the max_alpha and best_edge
                max_alpha_b_e = sum_o
                best_e = -1 * (self.env.regs.keys().index(key) + 1)  # region 0 will map to edge -1
                opt = (sum_o.T * np.matrix(b)).item(0)

        return max_alpha_b_e, best_e, opt

    def plot_bel(self, n, i, fig_n=1, b=None):
        """
        Give a one D plot of the value function at a given node by varying the uncertainty of the given region.
        Not implemented !

        :param n: n (q=node of DFA ,v= node of FIRM) in the
        :param i: The region key.
        :param b: list of probabilities associated to each of the regions.
        If None (=Default) take the initial probabilities.
        :param fig_n: handle for axes on which the plot can be given,
         if none automatically generate the plot. Otherwise add plot to the handle.
        :return: Returns a plot"""

        (q, v) = n
        vals = np.matrix([[]] * self.val[(q, v)].alpha_mat.shape[1])

        for b_i in np.linspace(0, 1, 20):
            b_ind = [prob if not el == i else [b_i]
                     for el, prob in enumerate(self.env.b_reg_init.tolist())]
            b = self.env.get_product_belief(b_ind)
            vals = np.concatenate((vals, self.val[(q, v)].alpha_mat.T * b), axis=1)

        fig = plt.figure(fig_n)
        for j in range(self.val[(q, v)].alpha_mat.shape[1]):
            plt.plot(np.linspace(0, 1, 20), vals[j,:].T)

        plt.ylabel('probability')
        plt.xlabel('b_' + self.env.regs.keys()[i])

        ax = plt.gca()
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.1, 1.1)
        #fig.show()


def bfs(graph, start):
    """
    Breadth first list of ordered nodes.

    :param graph: FIRM-type of graph
    :type graph: Spec_Spaths
    :param start: starting node for search
    :type start: tuple()
    :return: list of nodes
    """

    visited, queue = [], [start]
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited += [vertex]
            add_to = list((set(graph.predecessors(vertex)) - set(visited)) - set(queue))
            queue += add_to

    return visited


def optimizers(prod, ax, showplot=True, minimal=False):
    """
    Give a plot of the optimizers of the current graph.
    Based on the firm graph, but nodes now include also info on the state of the DFA

    :param showplot:
    :param prod: product of DFA and FIRM
    :type prod: Spec_Spaths
    :param ax: axes object
    :return: returns a plot
    """
    # build the nodes from the once that are actually reachable:
    # start from the initial node
    nodes = dict()  # empty dict with nodes as keys and values: (x,y),
    obs = dict()  # empty dict with nodes as keys, values: set of obs actions
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
        obs_actions = filter(lambda i: i < 0, opt)  # decide to observe a neigborhood
        tr_actions = filter(lambda i: i >= 0, opt)  # decide to transition to a new node
        # find (x,y)
        x = np.ravel(i_v.mean)[0]
        y = np.ravel(i_v.mean)[1]
        nodes[i_v] = (x, y)
        obs[i_v] = obs.get(i_v, set()) | set(obs_actions)
        edges |= {(i_v, i_next) for i_next in tr_actions}
        # to find the nodes (composed of i_q,i_v) that have not yet been added and
        # that are accessible from this node based ont he transition actions, check transitions in prod
        for n_next in prod[n]:
            if n_next[1] in tr_actions:
                if (n_next not in unvisited) and (not n_next in visited):
                    unvisited.extend([n_next])
    if showplot:
        for node in nodes:
            ax.plot(nodes[node][0], nodes[node][1], 'b')

        for (start, dest) in edges:
            plt.plot([nodes[start][0], nodes[dest][0]],
                     [nodes[start][1], nodes[dest][1]], color='black')

            plt.arrow(nodes[start][0], nodes[start][1],
                      .7 * (nodes[dest][0] - nodes[start][0]),
                      .7 * (nodes[dest][1] - nodes[start][1]),
                      head_width=0.2, head_length=.2,
                      fc='k', ec='k')

    if minimal:
        raise NotImplementedError

    return nodes, edges, visited


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
    q_ = None
    b_ = None
    for t in range(time_n):
        print(t)
        # Get best edge
        alpha_new, best_e, opt = spath._back_up(q, v, b)
        vals += [opt]

        if obs_action is True and best_e < 0:
            reg_key = spath.env.regs.keys()[-1 * (best_e + 1)]
            (b_, o, i_o) = spath.env.get_b_o_reg(b, spath.env.regs[reg_key][3], reg_key, v.mean)
            b = b_
            act_list += ["Observing = " + reg_key]
            print "Observing " + reg_key + " at vertex" + str(v) + " q = " + str(q)
            continue

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
        act_list += [best_e]

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
        print("going from vertex " + str(v) + " to vertex " + str(v_) + " q = " + str(q_))

        b = b_  # new value becomes old value
        q = q_  # new value becomes old value
        v = v_  # new value becomes old value
        v_list += [v]
        if q in list(spath.dfsa_final) or q == list(spath.dfsa_final):
            print('break')

            break

    return traj, v_list, vals, act_list
