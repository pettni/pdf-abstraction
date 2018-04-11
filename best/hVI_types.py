from collections import OrderedDict
import best.rss18_functions as rf
import numpy as np
import polytope as pc
from numpy import linalg as LA
import matplotlib.pyplot as plt

''' TODO:
- Unit test get_O matrix
'''


class Gamma(object):
    ''' alpha_mat = np.matrix([n_belief_states x n_belief_points])
        belief_points = list(np.matrix([n_belief_states x 1]))
    '''
    def __init__(self, belief_points, alpha_mat=None):
        if alpha_mat:
            self.alpha_mat = alpha_mat
        else:
            self.alpha_mat = np.zeros([len(belief_points[0]), len(belief_points)])
        self.belief_points = belief_points
        self.edge = []  #Used to store index of edge

    def prune(self):
        print "TODO: implement prune alpha"


# Belief MDP of environment
class Env(object):
    def __init__(self, regs):
        self.reg_index = {}  # saves the position of the region in observation vector
        self.regs = {}
        # add only unknown regions (discard regs with probability 0 or 1)
        for key, value in regs.iteritems():
            if (value[1] < 1.0) and (value[1] > 0.0):
                self.regs[key] = value
                self.reg_index[key] = self.regs.keys().index(key)
        self.n_unknown_regs = len(self.regs)
        self.n_total_regs = len(regs)
        # construct the product state space 2^n_unknown_regs
        self.x_e = [i for i in range(2**self.n_unknown_regs)]

    ''' returns O matrix
        v_mean = mean value of a FIRM node '''
    def get_O(self, v_mean):
        false_rate_regs = [self.get_false_rate(val, v_mean) for key, val in self.regs.iteritems()]
        O = np.ones([2**self.n_unknown_regs, 2**self.n_unknown_regs])
        for i_obs in range(len(self.x_e)):
            for i_x in range(len(self.x_e)):
                for i_reg in range(self.n_unknown_regs):
                    if self.x_e[i_obs] & 2**i_reg == self.x_e[i_x] & 2**i_reg:
                        O[i_obs][i_x] = O[i_obs][i_x] * (1-false_rate_regs[i_reg])
                    else:
                        O[i_obs][i_x] = O[i_obs][i_x] * false_rate_regs[i_reg]
        return O

    ''' Returns the false rate
        reg = value for RSS dictionary regs
        v_mean = mean value of a FIRM node '''
    def get_false_rate(self, reg, v_mean):
        bb = pc.bounding_box(reg[0])
        center = [(bb[0][0]+bb[1][0])/2, (bb[0][1]+bb[1][1])/2]
        # import pdb; pdb.set_trace()
        dist = LA.norm(center - v_mean)
        return self.get_false_rate_dist(dist)

    ''' Returns the false rate
        dist = distance between the rover and the region '''
    def get_false_rate_dist(self, dist):
        thresh1 = 1
        thresh2 = 5
        if dist < thresh1:
            return 0.0
        elif dist < thresh2:
            return (0.0 * (thresh2 - (dist-thresh1)) + 0.5 * (dist - thresh1))/(thresh2-thresh1)
        else:
            return 0.5


if __name__ == '__main__':
    ''' Define Regions and create Env '''
    regs = OrderedDict()
    # p1 = rf.vertex_to_poly(np.array([[1.2, 0], [2.2, 1], [-1.6, 3.6], [-2.6, 2.6]]))
    p1 = rf.vertex_to_poly(np.array([[-1, -2], [-1, 2], [1, 2], [1, -2]]))
    regs['r1'] = (p1, 1, 'red')
    p2 = rf.vertex_to_poly(np.array([[-3, 4], [-3, 5], [-5, 5], [-5, 4]]))
    regs['r2'] = (p2, 1, 'blue')
    # p3 = rf.vertex_to_poly(np.array([[2, -1.5], [3, -1], [5, -3], [5, -5], [4, -5]]))
    # regs['r3'] = (p3, 1, 'red')
    env = Env(regs)
    O = env.get_O(np.array([1,1]))
    print O
    ''' Plot False Rate vs Distance '''
    dist = np.arange(0.0, 8.0, 0.1)
    false_rate = np.array([env.get_false_rate_dist(d) for d in dist])
    plt.plot(dist, false_rate, 'r')
    plt.xlabel('Distance (m)')
    plt.ylabel('False Rate')
    plt.show()
