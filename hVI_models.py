import math
import random
from itertools import product

import numpy as np
from numpy import linalg as LA
from scipy.linalg import solve_discrete_are as dare


# Define Belief Space/State
class Belief_Space(object):

    def distance(self, belief_state_1, belief_state_2):
        raise NotImplementedError

    def distance_mean(self, belief_state_1, belief_state_2):
        raise NotImplementedError

    def new_state(self, mean, cov):
        raise NotImplementedError


class State_Space(object):
    """
    State_space() is a class object for the
    state space of the system that allows for the drawing of samples needed
    for the sampling-based roadmaps
    """

    def __init__(self, x_low, x_up):
        self.x_low = x_low
        self.x_up = x_up
        self.grid = None

    def sample_new_state_from_grid(self):
        """ Returns a random state  """
        if not self.grid:
            gridxy = [[self.x_low[i] + (self.x_up[i] - self.x_low[i]) * (l + 1) / 8 for l in range(8)] for i in
                      range(len(self.x_low))]
            self.grid = list(product(*gridxy))
            print(self.grid)

            random.shuffle(self.grid)
        mean = self.grid.pop()
        print(mean)
        return State(mean)

    def sample_new_state(self):
        """ Returns a random state  """
        mean = [self.x_low[i] + (self.x_up[i] - self.x_low[i]) * np.random.rand()
                for i in range(len(self.x_low))]
        return State(mean)

    def sample_new_state_in_reg(self, reg_polytope):
        x_low = reg_polytope.bounding_box[0].ravel()
        x_up = reg_polytope.bounding_box[1].ravel()
        mean = [x_low[i] + (x_up[i] - x_low[i]) * 0.5
                for i in range(len(x_low))]
        return State(mean)

    def distance(self, state1, state2):

        return self.distance_mean(state1, state1)

    def distance_mean(self, state1, state2):
        if isinstance(state1, State):
            mean1 = state1.mean
        else:
            mean1 = np.mat(state1)
            if mean1.shape[0] == 1:
                mean1 = mean1.T

        if isinstance(state2, State):
            mean2 = state2.mean
        else:
            mean2 = np.mat(state2)
            if mean2.shape[0] == 1:
                mean2 = mean2.T

        return LA.norm(mean1 - mean2)

    def new_state(self, mean):
        if isinstance(mean, State):
            return mean
        else:
            return State(mean)


class Rn_Belief_Space(Belief_Space):

    def __init__(self, x_low, x_up):
        self.x_low = x_low
        self.x_up = x_up

    def sample_new_state(self):
        """ Returns a belief state with random mean and zero covariance """
        mean = np.array([[self.x_low[i] + (self.x_up[i] - self.x_low[i]) * np.random.rand()]
                         for i in range(len(self.x_low))])
        return Rn_Belief_State(mean)

    def sample_new_state_in_reg(self, reg_polytope):
        x_low = reg_polytope.bounding_box[0].ravel()
        x_up = reg_polytope.bounding_box[1].ravel()
        mean = [x_low[i] + (x_up[i] - x_low[i]) * 0.5
                for i in range(len(x_low))]
        return Rn_Belief_State(mean)

    def distance(self, belief_state1, belief_state2):
        distance_cov = LA.norm(belief_state1.cov - belief_state2.cov)
        return self.distance_mean(belief_state1, belief_state2) + 0.1 * distance_cov

    def distance_mean(self, belief_state1, belief_state2):
        return LA.norm(belief_state1.mean - belief_state2.mean)

    def new_state(self, mean, cov=None):
        return Rn_Belief_State(mean, cov)


class Belief_State(object):

    def __init__(self, mean, cov=None):
        pass


class Rn_Belief_State(Belief_State):

    def __init__(self, mean, cov=None):
        super(Rn_Belief_State, self).__init__(mean, cov)
        self.mean = np.mat(mean)
        if self.mean.shape[0] == 1:
            self.mean = self.mean.T
        if cov is None:
            self.cov = np.zeros([len(mean), len(mean)])
        else:
            self.cov = np.mat(cov)


class State(Belief_State):
    """The state space of a deterministic model"""

    def __init__(self, mean, cov=None):
        super(State, self).__init__(mean, cov)
        self.mean = np.mat(mean)
        if self.mean.shape[0] == 1:
            self.mean = self.mean.T
        if cov is None:
            self.cov = np.zeros([len(mean), len(mean)])
        else:
            raise ValueError

    def __str__(self):
        return self.mean.__str__()

    def __repr__(self):
        return "<node at x:%s y:%s>" % (np.ravel(self.mean)[0], np.ravel(self.mean)[1])


# Define Motion Models
class Motion_Model(object):
    """ x_k+1 = f(x_k, u_k, v_k)  """

    def getA(self, belief_state):
        raise NotImplementedError

    def getB(self, belief_state):
        raise NotImplementedError

    def getG(self, belief_state):
        raise NotImplementedError

    def getQ(self, belief_state):
        raise NotImplementedError

    def generate_noise(self, belief_state):
        raise NotImplementedError

    def generate_desiredtraj_and_ffinput(self, node_i, node_j):
        raise NotImplementedError

    def evolve(self, b, u, w):
        raise NotImplementedError


class Det_SI_Model(Motion_Model):
    """ This is a simple single integrator model:
    [[x],[y]] = [[1, 0],[0,1]] [[x],[y]] + [[1, 0],[0,1]] u (in continuous time)
    """

    def __init__(self, dt, statespace=State_Space([-1, -1], [1, 1])):
        """ Initialize the deterministic dynamic model

    :param dt: time discretization of this original continuous model
    :param statespace: Statespace of the mode
    :type statespace: State_Space
    """
        self.dt = dt
        self.max_speed = 0.5
        self.A = np.eye(2)
        self.B = dt * np.eye(2)
        self.state_space = statespace

        self.Ls = None
        self.Wx = None
        self.Wu = None
        print(" The used model contains an integrator for each dimension:\n")
        print(str(self.A))

    def getA(self, belief_state):
        return self.A

    def getB(self, belief_state):
        return self.B

    def getLS(self, Wx, Wu):
        if self.Ls is None:
            print(self.A)
            print(self.B)
            self.Wx = Wx
            self.Wu = Wu
            print(dare(self.A, self.B, self.Wx, self.Wu))
            S = np.mat(dare(self.A, self.B, self.Wx, self.Wu))
            self.Ls = (self.B.T * S * self.B + self.Wu).I * self.B.T * S * self.A

            return self.Ls
        else:
            if (self.Wx == Wx).all() & (self.Wu == Wu).all():
                return self.Ls
            else:
                self.Wx = Wx
                self.Wu = Wu
                S = np.mat(dare(self.A, self.B, self.Wx, self.Wu))
                self.Ls = (self.B.T * S * self.B + self.Wu).I * self.B.T * S * self.A

                return self.Ls

    def generate_desiredtraj_and_ffinput(self, node_i, node_j):
        # NOTE: ignoring covariance
        N = int(math.floor(LA.norm(node_j.mean - node_i.mean) / (self.max_speed * self.dt)))
        traj_d = [node_i.mean]
        u_ff = []
        for k in range(1, N + 1):
            traj_d_k = node_i.mean + k * (node_j.mean - node_i.mean) / N
            traj_d.append(traj_d_k)
            speed_k = LA.norm(traj_d[-2] - traj_d[-1]) / self.dt
            u_ff_k = speed_k * (node_j.mean - node_i.mean) / LA.norm(node_j.mean - node_i.mean)
            u_ff.append(u_ff_k)
        return traj_d, u_ff

    def evolve(self, b, u):
        bnew = self.state_space.new_state(self.A * b.mean + self.B * u)
        return bnew


class SI_Model(Motion_Model):
    """ Single Integrator Model """

    def __init__(self, dt):
        self.dt = dt
        self.max_speed = 0.5
        self.A = np.eye(2)
        self.B = dt * np.eye(2)
        self.G = np.eye(2)
        self.Q = 0.001 * np.eye(2)
        self.belief_space = Rn_Belief_Space([-1, -1], [1, 1])

        print(" The used model contains an integrator for each dimension:\n")
        print(str(self.A))

    def getA(self, belief_state):
        return self.A

    def getB(self, belief_state):
        return self.B

    def getG(self, belief_state):
        return self.G

    def getQ(self, belief_state):
        return self.Q

    def generate_noise(self, belief_state, control):
        return np.matrix(np.random.multivariate_normal(mean=[0, 0], cov=self.Q)).T

    def generate_desiredtraj_and_ffinput(self, node_i, node_j):
        # NOTE: ignoring covariance
        N = int(math.floor(LA.norm(node_j.mean - node_i.mean) / (self.max_speed * self.dt)))
        traj_d = [node_i]
        u_ff = []
        for k in range(1, N + 1):
            traj_d_k = node_i.mean + k * (node_j.mean - node_i.mean) / N
            traj_d.append(self.belief_space.new_state(traj_d_k, np.zeros([2, 2])))
            speed_k = LA.norm(traj_d[-2].mean - traj_d[-1].mean) / self.dt
            u_ff_k = speed_k * (node_j.mean - node_i.mean) / LA.norm(node_j.mean - node_i.mean)
            u_ff.append(u_ff_k)
        return traj_d, u_ff

    def evolve(self, b, u, w):
        bnew = self.belief_space.new_state(self.A * b.mean + self.B * u + w, b.cov)
        return bnew


# Define Observation Models
class Observation_Model(object):
    """ z = h(x) + N(0,R)"""

    def getH(self, belief_state):
        raise NotImplementedError

    def getM(self, belief_state):
        raise NotImplementedError

    def getR(self, belief_state):
        raise NotImplementedError

    def get_obs_prediction(self, belief_state):
        raise NotImplementedError

    def get_obs(self):
        raise NotImplementedError


class Gaussian_Noise(Observation_Model):
    """ Gaussian Noise in dim dimensions"""

    def __init__(self, dim):
        self.dim = dim
        self.H = np.eye(dim)
        self.M = np.eye(dim)
        self.R = 0.1 * np.eye(dim)

    def getH(self, belief_state):
        return self.H

    def getM(self, belief_state):
        return self.M

    def getR(self, belief_state):
        return self.R

    def get_obs_prediction(self, belief_state):
        return belief_state.mean

    def get_obs(self, belief_state):
        return belief_state.mean + np.matrix(
            np.random.multivariate_normal(mean=[0 for _ in range(self.dim)], cov=self.R)).T

# motion_model = SI_Model(0.5)
# # Test ffinput of SI_Model
# node_i = motion_model.belief_space.new_state(np.zeros([2,1]),np.zeros([2,2]))
# node_j = motion_model.belief_space.new_state(np.ones([2,1]),np.zeros([2,2]))
# A = motion_model.getA(0)
# B = motion_model.getB(0)
# [traj_d, u] = motion_model.generate_desiredtraj_and_ffinput(node_i, node_j)
# traj = [traj_d[0]]
# for i in range(len(u)):
#     x = A * np.mat(traj[-1].mean) + B * np.mat(u[i])
#     traj.append(motion_model.belief_space.new_state(x,np.zeros([2,2])))
#     print "Error = "
#     print traj_d[i+1].mean - x
# obs_model = Gaussian_Noise(4)
# print motion_model.A
# print motion_model.B
# print obs_model.H
# import pdb; pdb.set_trace()
