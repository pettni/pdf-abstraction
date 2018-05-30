from collections import OrderedDict
import best.rss18_functions as rf
import numpy as np
import polytope as pc
from numpy import linalg as LA
import matplotlib.pyplot as plt
import random
from hVI_config import obs_action

''' TODO:
- Unit test get_O matrix
- Unit test get_b_o
'''

class Gamma(object):
    ''' b_prod_points = list(np.matrix([n_belief_states x 1]))
        alpha_mat = np.matrix([n_belief_states x n_belief_points])'''
    def __init__(self, b_prod_points, b_reg_points=None, alpha_mat=None):
        if alpha_mat:
            self.alpha_mat = np.matrix(alpha_mat)
        else:
            self.alpha_mat = np.matrix(np.zeros([len(b_prod_points[0]), len(b_prod_points)]))
        self.b_prod_points = b_prod_points
        self.best_edge = -np.ones(len(b_prod_points), dtype=np.int8)  # policy
        if obs_action is True:
            if b_reg_points is None:
                raise ValueError("b_reg_points cannot be None for when obs_action is True")
            else:
                self.b_reg_points = b_reg_points

    def prune(self):
        print "TODO: implement prune alpha"


# Belief MDP of environment
class Env(object):
    ''' Naming conventions:
    _reg = in lower dimensional space (prob of label = 1)
    _prod = in product space '''
    def __init__(self, regs):
        self.reg_index = {}  # saves the position of the region in observation vector
        self.regs = {} # stores only unknown regs
        self.b_reg_init = []
        # add only unknown regions (discard regs with probability 0 or 1)
        for key, value in regs.iteritems():
            if (value[1] < 1.0) and (value[1] > 0.0):
                self.regs[key] = value
                self.reg_index[key] = self.regs.keys().index(key)
                self.b_reg_init.append(value[1])
        self.n_unknown_regs = len(self.regs)
        self.n_total_regs = len(regs)
        # construct the product state space 2^n_unknown_regs
        self.x_e = [i for i in range(2**self.n_unknown_regs)]
        self.b_reg_init = np.matrix(self.b_reg_init).T
        self.b_prod_init = self.get_product_belief(self.b_reg_init)

    ''' returns O matrix (2^n_unknown_regs x 2^n_unknown_regs)
        v_mean = mean value of a FIRM node '''
    def get_O_prod(self, v_mean):
        regs = self.regs
        false_rate_regs = [self.get_false_rate(val, v_mean) for key, val in regs.iteritems()]
        O = np.matrix(np.ones([2**self.n_unknown_regs, 2**self.n_unknown_regs]))
        for i_obs in range(len(self.x_e)):
            for i_x in range(len(self.x_e)):
                for i_reg in range(self.n_unknown_regs):
                    if self.x_e[i_obs] & 2**i_reg == self.x_e[i_x] & 2**i_reg:
                        O[i_obs, i_x] = O[i_obs, i_x] * (1-false_rate_regs[i_reg])
                    else:
                        O[i_obs, i_x] = O[i_obs, i_x] * false_rate_regs[i_reg]
        return O

    ''' returns [p(o_reg=True|v_mean); p(o_reg=False|v_mean)]  (2 x 2^n_unknown_regs)
        v_mean = mean value of a FIRM node '''
    def get_O_reg_prob(self, v_mean, reg_key):
        false_rate = self.regs[reg_key]
        O = np.matrix(np.ones([2, 2**self.n_unknown_regs]))
        # TODO: Validate that x_e is ordered by index of reg
        i_obs = self.regs.keys().index(reg_key)
        for i_x in range(len(self.x_e)):
            if self.x_e[i_x] & 2**i_obs == 2**i_obs:
                O[0, i_x] = O[0, i_x] * (1-false_rate)
                O[1, i_x] = O[1, i_x] * (false_rate)
            else:
                O[0, i_x] = O[1, i_x] * (false_rate)
                O[1, i_x] = O[0, i_x] * (1-false_rate)
        O[0, i_x] = O[0, i_x] / sum(O[0, :])
        O[1, i_x] = O[1, i_x] / sum(O[1, :])
        return O

    ''' returns O matrix (2 x 2)
        v_mean = mean value of a FIRM node
        reg_key = key for a particular region '''
    def get_O_reg(self, v_mean, reg_key):
        false_rate = self.get_false_rate(self.regs[reg_key], v_mean)
        O = np.matrix([[1-false_rate, false_rate],
                      [false_rate, 1-false_rate]])
        return O

    def get_b_o_prod(self, v_mean, b, x_e_true):
        O = self.get_O_prod(v_mean)
        p_o = np.ravel(O[:, np.argmax(x_e_true)]).tolist()
        n_rand = random.random()
        p_cum = 0
        for p in p_o:
            p_cum = p_cum + p
            if n_rand < p_cum:
                i_o = p_o.index(p)
                break
        o = self.x_e[i_o]
        b_ = np.multiply(O[i_o, :].T, b)/(O[i_o, :] * b)
        return (b_, o, i_o)

    ''' Returns the false rate
        reg = value for RSS dictionary regs
        v_mean = mean value of a FIRM node '''
    def get_false_rate(self, reg, v_mean):
        bb = pc.bounding_box(reg[0])
        center = [(bb[0][0]+bb[1][0])/2, (bb[0][1]+bb[1][1])/2]
        dist = LA.norm(center - v_mean)
        return self.get_false_rate_dist(dist)

    ''' Returns the false rate
        dist = distance between the rover and the region '''
    def get_false_rate_dist(self, dist):
        thresh1 = 1
        thresh2 = 5
        if dist < thresh1:
            return 0.2
        elif dist < thresh2:
            return (0.0 * (thresh2 - (dist-thresh1)) + 0.5 * (dist - thresh1))/(thresh2-thresh1)
        else:
            return 0.5

    ''' Returns the belief vector in product belief space given belief of
        each individual reg p(x_ei == 1) '''
    def get_product_belief(self, belief):
        if type(belief) is list:
            belief = np.matrix(belief)
        if belief.shape[0] == 1:
            belief = belief.T
        if belief.shape[0] is not self.n_unknown_regs:
            raise ValueError('Size of belief should be equal to n_unknown_regs')
        b_prod = np.matrix(np.zeros([len(self.x_e), 1]))
        for i in range(len(self.x_e)):
            b_prod[i, 0] = 1
            for j in range(self.n_unknown_regs):
                if self.x_e[i] & 2**j == 2**j:
                    b_prod[i, 0] = b_prod[i, 0] * belief[j]
                else:
                    b_prod[i, 0] = b_prod[i, 0] * (1-belief[j])
        return b_prod

    ''' Returns the belief vector in product belief space given belief of
        each individual reg p(x_ei == 1) '''
    def get_reg_belief(self, belief):
        if type(belief) is list:
            belief = np.matrix(belief)
        if belief.shape[0] == 1:
            belief = belief.T
        if belief.shape[0] is not self.n_unknown_regs:
            raise ValueError('Size of belief should be equal to n_unknown_regs')
        return belief

if __name__ == '__main__':
    ''' Define Regions and create Env '''
    regs = OrderedDict()
    # p1 = rf.vertex_to_poly(np.array([[1.2, 0], [2.2, 1], [-1.6, 3.6], [-2.6, 2.6]]))
    p1 = rf.vertex_to_poly(np.array([[-1, -2], [-1, 2], [1, 2], [1, -2]]))
    regs['r1'] = (p1, 0.5, 'red')
    p2 = rf.vertex_to_poly(np.array([[-3, 4], [-3, 5], [-5, 5], [-5, 4]]))
    regs['r2'] = (p2, 0.5, 'blue')
    # p3 = rf.vertex_to_poly(np.array([[2, -1.5], [3, -1], [5, -3], [5, -5], [4, -5]]))
    # regs['r3'] = (p3, 1, 'red')
    env = Env(regs)
    O = env.get_O(np.array([1,1]))
    print O
    ''' Plot False Rate vs Distance '''
    dist = np.arange(0.0, 8.0, 0.1)
    false_rate = np.array([env.get_false_rate_dist(d) for d in dist])
    b = [0.1, 0.1]
    b_prod = env.get_product_belief(b)
    print b_prod
    plt.plot(dist, false_rate, 'r')
    plt.xlabel('Distance (m)')
    plt.ylabel('False Rate')
    plt.show()
