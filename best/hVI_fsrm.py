'''
FSRM = Feedback State RoadMap
This file gives the class for the generation of a sampling based path planner
that can be used together with the HVI tools.
'''

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

    def make_nodes_edges(self, number, edges ):
        t = [time.clock()]
        self.sample_nodes(number)
        t += [time.clock()]

        self.make_edges(edges)
        t += [time.clock()]

        self.n_particles = 1
        print(np.diff(t))

    def sample_nodes(self, n_nodes, means=list(), append=False):
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

                if i < len(self.regs) and ((self.regs[self.regs.keys()[i]][2] is not 'obs') or (self.regs[self.regs.keys()[i]][1]<1)):
                        node = self.state_space.sample_new_state_in_reg(self.regs[self.regs.keys()[i]][0])
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

    ''' Returns a set of outputs generated by the trajectory.
    For now it returns only one output with the lowest position in the dictionary,
    even if multiple outputs are generated '''
    # traj = list of belief_state(s)
    # [R] output = key of regs
    def get_outputs(self, src, dest=None):
        output = ['null']
        start = (src.mean if isinstance(src, State) else src)

        if dest:

            end = (dest.mean if isinstance(dest,State) else dest)


            for (name, info) in self.regs.iteritems():
                if name == 'null':
                    continue
                box = info[0]
                if self.intersect(box, start, end):
                    output += [name]
                    if self.ax is not None:
                        self.plot_traj([src] + [dest], self.output_color[name])
            return output

        else: # this is if dest is None
            for (name, info) in self.regs.iteritems():
                if name == 'null':
                    continue
                box = info[0]
                if box.contains(start):
                    output = [name]

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
            #self.ax.plot(x, y, color, ms=20,linewidth=3.0)

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
        traj = [b0.mean]
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

