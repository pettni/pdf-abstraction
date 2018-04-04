from best.mdp import MDP
from models import Gaussian_Noise, SI_Model, Rn_Belief_Space, Rn_Belief_State
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.linalg import solve_discrete_are as dare
import itertools as it
from collections import OrderedDict
import best.rss18_functions as rf

'''
List of pending improvements:
- Update omniwheel model
- Improve collision checking strategy
- Verify LQG implementation
'''

class FIRM(object):

    def __init__(self, belief_space, motion_model, obs_model, Wx, Wu, regs, regs_outputs, output_color, ax):
        ''' Create a FIRM graph with n_nodes in [x_low, x_up]; informed sampling in regs '''
        self.belief_space = belief_space
        self.motion_model = motion_model
        self.obs_model = obs_model
        # np.random.seed(12)
        self.regs = regs
        self.Wx = Wx
        self.Wu = Wu
        self.regs_outputs = regs_outputs
        self.output_color = output_color
        self.ax = ax
        self.nodes = []
        self.node_controllers = []
        self.edges = {}
        self.edge_controllers = {}
        self.edge_output_prob = {}
        self.T_list = None
        self.sample_nodes(20)
        self.make_edges(4)
        # assert np.all(np.linalg.eigvals(self.state_weight) > 0) # check pdef
        # assert np.all(np.linalg.eigvals(self.control_weight) > 0) # check
        # pdef
        self.n_particles = 2

    def abstract(self):
        ''' Construct T by treating index of neigh as action number '''
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
        import pdb; pdb.set_trace()
        return MDP(self.T_list, output_name='xc', output_fcn=output_fcn)

    def sample_nodes(self, n_nodes, append=False):
        ''' Generate n_nodes Nodes by randomly sampling in [x_low, x_up] '''
        # TODO: Implement append to sample nodes incrementally
        if append is False:
            self.nodes = []  # clear previous nodes/edges
            self.edges = {}
            self.node_controllers = []
            self.edge_controllers = {}
        for i in range(n_nodes):
            # Sample Mean
            node = self.belief_space.sample_new_state()
            # Set Co-variance
            A = self.motion_model.getA(node)
            G = self.motion_model.getG(node)
            Q = self.motion_model.getQ(node)
            H = self.obs_model.getH(node)
            M = self.obs_model.getM(node)
            R = self.obs_model.getR(node)
            Pprd = np.mat(dare(A.T, H.T, G * Q * G.T, M * R * M.T))
            assert np.all(np.isreal(Pprd)) and np.all(Pprd == Pprd.T)
            Pest = Pprd - (
                Pprd * H.T) * (
                H * Pprd * H.T + M * R * M.T).I * (
                Pprd * H.T).T
            assert np.all(np.isreal(Pest)) and np.all(Pest == Pest.T)
            node.cov = Pest
            self.nodes.append(node)
            self.node_controllers.append(
                Node_Controller(self.motion_model, self.obs_model,
                                node, self.Wx, self.Wu,
                                self.belief_space))
        # TODO:''' Generate one node in each region '''
        # j=0
        # for key in self.regs:
        #   x_low = self.regs[key][0].bounding_box[0].ravel()
        #   x_up = self.regs[key][0].bounding_box[1].ravel()
        #   for i in range(len(self.x_low)):
        #     self.nodes[n_nodes+j,i] = x_low[i] + (x_up[i] - x_low[i])*np.random.rand(1).ravel()
        #   j=j+1

    def make_edges(self, dist):
        ''' Construct edges for self.nodes within distance dist '''
        self.max_actions = 1  # '''used to construct T'''
        # Can make more efficient
        for i in range(len(self.nodes)):
            neigh = []
            edge_controllers = []
            for j in range(len(self.nodes)):
                dist_nodes = self.belief_space.distance_mean(self.nodes[i], self.nodes[j])
                if dist_nodes < dist and dist_nodes > 0:
                    neigh.append(j)
                    edge_controllers.append(Edge_Controller(self.motion_model,
                                                            self.obs_model, self.nodes[
                                                            i], self.nodes[
                                                                j], self.Wx,
                                                            self.Wu, self.belief_space))
            if len(neigh) > self.max_actions:
                self.max_actions = len(neigh)
            self.edges[i] = neigh
            self.edge_controllers[i] = edge_controllers

    def compute_output_prob(self):
        ''' Compute the probability over output set for each edge in the FIRM graph '''
        # traj2 = self.node_controllers[self.edges[self.edges.keys()[0]][0]].simulate_trajectory(traj1[-1])
        # output2 = self.get_outputs(traj2)
        for (node, edge_controllers) in self.edge_controllers.iteritems():
            output_prob_edges = []
            for edge_controller in edge_controllers:
                for i in range(self.n_particles):
                    p_out = np.zeros([len(self.regs_outputs), 1])
                    traj = edge_controller.simulate_trajectory()
                    output = self.get_outputs(traj)
                    p_out[output] = p_out[output] + 1
                    if self.ax is not None:
                        self.plot_traj(traj, self.output_color[output])
                p_out = p_out/self.n_particles
                output_prob_edges.append(p_out)
            self.edge_output_prob[node] = output_prob_edges

    def plot_traj(self, traj, color):
        for i in range(len(traj)-1):
            x = [np.ravel(traj[i].mean)[0], np.ravel(traj[i+1].mean)[0]]
            y = [np.ravel(traj[i].mean)[1], np.ravel(traj[i+1].mean)[1]]
            self.ax.plot(x, y, color, ms=20)

    def plot(self, ax):
        ''' Plot the FIRM graph '''
        for i in range(len(self.nodes)):
            try:
                neigh = self.edges[i]
            except KeyError:
                continue
            for j in neigh:
                x = [np.ravel(self.nodes[i].mean)[0], np.ravel(self.nodes[j].mean)[0]]
                y = [np.ravel(self.nodes[i].mean)[1], np.ravel(self.nodes[j].mean)[1]]
                # ax.plot(x, y, 'b')
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
        # plt.set_xlim(self.belief_space.x_low[0], self.belief_space.x_up[0])
        # plt.set_ylim(self.belief_space.x_low[1], self.belief_space.x_up[1])
        for (name, info) in self.regs.iteritems():
            hatch = False
            fill = True
            rf.plot_region(ax, info[0], name, 1, self.output_color[self.regs_outputs[info[2]]], hatch=hatch, fill=fill)

    def get_outputs(self, traj):
        ''' Returns a set of outputs generated by the trajectory.
            For now it returns only one output with the highest number,
            even if multiple outputs are generated '''
        output = -1
        for belief in traj:
            for (name, info) in regs.iteritems():
                poly = info[0]
                if poly.contains(belief.mean):
                    if output < self.regs_outputs[info[-1]]:
                        output = self.regs_outputs[info[-1]]
        return output


class Node_Controller(object):

    ''' Consists of SLGR and SKF '''
    # Wx, Wu = state and control error weights for LQR

    def __init__(self, motion_model, obs_model, node, Wx, Wu, belief_space):
        self.motion_model = motion_model
        self.obs_model = obs_model
        self.A = self.motion_model.getA(node)
        self.B = self.motion_model.getB(node)
        self.G = self.motion_model.getG(node)
        self.Q = self.motion_model.getQ(node)
        self.H = self.obs_model.getH(node)
        self.M = self.obs_model.getM(node)
        self.R = self.obs_model.getR(node)
        self.node = node
        self.Wx = Wx
        self.Wu = Wu
        self.belief_space = belief_space
        # set control gain
        S = np.mat(dare(self.A, self.B, self.Wx, self.Wu))
        self.Ls = (self.B.T * S * self.B + self.Wu).I * self.B.T * S * self.A

    def simulate_trajectory(self, b0):
        ''' Simulates trajectory form node_i to node_j '''
        traj = [b0]
        while not self.belief_space.distance(traj[-1], self.node) < 0.1:
            ''' Get control, apply control, get observation and apply observation '''
            b = traj[-1]
            # Get control
            u_k = -self.Ls * (b.mean - self.node.mean)
            # Apply control/predict (Esq. 75-76, 98-99)
            # x^-_k+1 = A * x^+_k + B * u; P^-_k+1 = A * P^+_k *A^T + G Q G^T
            w = self.motion_model.generate_noise(b, u_k)
            bnew_pred = self.motion_model.evolve(b, u_k, w)
            bnew_pred.cov = self.A * b.cov * \
                self.A.T + self.G * self.Q * self.G.T
            # Get measurement
            z = self.obs_model.get_obs(bnew_pred)
            # Update Belief
            innovation = z - self.obs_model.get_obs_prediction(bnew_pred)
            K = bnew_pred.cov * self.H.T * \
                (self.H * bnew_pred.cov * self.H.T +
                 self.M * self.R * self.M.T).I
            bnew_mean = bnew_pred.mean + K * innovation
            bnew_cov = bnew_pred.cov - K * self.H * bnew_pred.cov
            # Append traj
            traj.append(self.belief_space.new_state(bnew_mean, bnew_cov))
        return traj


class Edge_Controller(object):
    ''' Time varying LQG controller '''
    def __init__(self, motion_model, obs_model, node_i, node_j, Wx, Wu,
                 belief_space):
        self.motion_model = motion_model
        self.obs_model = obs_model
        self.node_i = node_i
        self.node_j = node_j
        self.Wx = Wx
        self.Wu = Wu
        self.belief_space = belief_space
        self.ax = ax
        [self.traj_d, self.u0] = self.motion_model.generate_desiredtraj_and_ffinput(
            node_i, node_j)
        self.N = len(self.traj_d)
        n_xdim = self.motion_model.getA(node_i).shape[1]
        n_udim = self.motion_model.getB(node_i).shape[1]
        # Generate feedback gains
        S = np.empty((self.N + 1, n_xdim, n_xdim), dtype=np.float)
        self.L = np.empty((self.N, n_udim, n_xdim), dtype=np.float)
        S[self.N, :, :] = self.Wx
        for t in it.count(self.N - 1, -1):
            A, B, S_next = map(np.mat, [self.motion_model.getA(self.traj_d[t]),
                                        self.motion_model.getB(self.traj_d[t]),
                                        S[t + 1, :, :]])
            L = (B.T * S_next * B + self.Wu).I * B.T * S_next * A
            self.L[t, :, :] = L
            S[t, :, :] = self.Wx + A.T * S_next * (A - B * L)
            if t == 0:
                break

    def simulate_trajectory(self):
        ''' Simulates trajectory starting from node_i to node_j '''
        traj = [self.node_i]
        for t in range(self.N-1):
            b = traj[-1]
            u = -self.L[t, :, :] * (b.mean - self.traj_d[t].mean) + self.u0[t]
            w = self.motion_model.generate_noise(b, u)
            bnew_pred = self.motion_model.evolve(b, u, w)
            A = self.motion_model.getA(b)
            G = self.motion_model.getG(b)
            Q = self.motion_model.getQ(b)
            H = self.obs_model.getH(b)
            M = self.obs_model.getM(b)
            R = self.obs_model.getR(b)
            bnew_pred.cov = A * b.cov * A.T + G * Q * G.T
            # Get measurement
            z = self.obs_model.get_obs(bnew_pred)
            # Update Belief
            innovation = z - self.obs_model.get_obs_prediction(bnew_pred)
            K = bnew_pred.cov * H.T * (H * bnew_pred.cov * H.T + M * R * M.T).I
            bnew_mean = bnew_pred.mean + K * innovation
            bnew_cov = bnew_pred.cov - K * H * bnew_pred.cov
            traj.append(self.belief_space.new_state(bnew_mean, bnew_cov))
        return traj

Wx = np.eye(2)
Wu = np.eye(2)
r2_bs = Rn_Belief_Space([-5, -5], [5, 5])
motion_model = SI_Model(0.1)
obs_model = Gaussian_Noise(2)
regs = OrderedDict()

p1 = rf.vertex_to_poly(np.array([[1.2, 0], [2.2, 1], [-1.6, 3.6], [-2.6, 2.6]]))
regs['r1'] = (p1, 1, 'red')
p2 = rf.vertex_to_poly(np.array([[-3, 4], [-3, 5], [-5, 5], [-5, 4]]))
regs['r2'] = (p2, 1, 'red')
p3 = rf.vertex_to_poly(np.array([[2, -1.5], [3, -1], [5, -3], [5, -5], [4, -5]]))
regs['r3'] = (p3, 1, 'red')
p4 = rf.vertex_to_poly(np.array([[1.2, 0], [2.2, 1], [2, -1.5], [3, -1]]))
regs['r4'] = (p4, 0.4, 'green')
p5 = rf.vertex_to_poly(np.array([[2, -1.5], [2.5, -2.5], [1, -5], [-1, -5]]))
regs['r5'] = (p5, 0.3, 'green')
a1 = rf.vertex_to_poly(np.array([[4, -2], [5, -2], [5, -1], [4, -1]]))
regs['a1'] = (a1, 0.5, 'blue')
a2 = rf.vertex_to_poly(np.array([[2, -4], [3, -4], [3, -5], [2, -5]]))
regs['a2'] = (a2, 0.5, 'blue')
a3 = rf.vertex_to_poly(np.array([[-2, 0], [-2, 1], [-1, 1], [-1, 0]]))
regs['a3'] = (a3, 0.9, 'blue')

# maps the name of a region to an output (number)
fig = plt.figure(0)
ax = fig.add_subplot(111, aspect='equal')
regs_output = {'blue':0, 'green':1, 'red':2}
output_color = {-1:'black', 0:'blue', 1:'green', 2:'red'}
firm = FIRM(r2_bs, motion_model, obs_model, Wx, Wu, regs, regs_output, output_color, ax)
firm.compute_output_prob()
firm.plot(ax)
plt.show()
# firm.abstract()
