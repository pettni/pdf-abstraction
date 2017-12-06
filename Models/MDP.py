""" Routines for handling  Markov Decision processes

author S. Haesaert
"""
from mdptoolbox.mdp import *
import numpy as np
import polytope as pc
import itertools

class Markov(MDP):

    def __init__(self,transitions, srep, urep,sedge):
        reward = np.ones((1,np.shape(transitions)[1])) # np.array( [[0.4487, 0.216, 0.2164, 0.1186, 0.000296],
              # [0.216, 0.1778, 0.3719, 0.2334, 0.0008789],
              # [0.09639, 0.1657, 0.6569, 0.08082, 0.0001928],
              # [0.005234, 0.0103, 0.008007, 0.9708, 0.005667],
              # [00, 0, 0, 0, 0.7788]])# 1#: array
            #Reward vectors.
        discount = None#: float
            #The discount rate on future rewards.
        max_iter = None#: int
            #The maximum number of iterations.
        epsilon = None#: tuple
            #The optimal policy.
        max_iter = None #: float
            #The time used to converge to the optimal policy.

        MDP.__init__(self, transitions, reward, discount, epsilon, max_iter,
                 skip_check=True)
        self.target = np.zeros((self.S,1))
        self.srep = srep
        self.sedge = sedge
        self.urep = urep
        self.V = np.zeros((self.S,1))
        self.policy_u = np.empty((self.S, len(self.urep)))

    def settarget(self, Target=None):
        if Target is None:
             return self.target

        if isinstance(Target,pc.Polytope):
            self.target= np.array([[1.] if pc.is_inside(Target, np.array(s)) else [0.] for s in itertools.product(*self.srep)])
            print(np.shape(self.target))
        else:
            self.target = Target


    def reach_bell(self, V = None):
        print('it')
        xi, yi = np.meshgrid(*self.srep)

        # Apply the Bellman operator on the value function.
        #
        # Updates the value function and the Vprev-improving policy.
        #
        # Returns: (policy, value), tuple of new policy and its value
        #
        # If V hasn't been sent into the method, then we assume to be working
        # on the objects V attribute
        if V is None:
            # this V should be a reference to the data rather than a copy
            V = self.V
        if self.policy  is None:
            self.policy = np.empty((1,self.S))

        else:
            # make sure the user supplied V is of the right shape
            try:
                assert V.shape in ((self.S,), (self.S,1)), "V is not the " \
                    "right shape (Bellman operator)."
            except AttributeError:
                raise TypeError("V must be a numpy array or matrix.")
        # Looping through each action the the Q-value matrix is calculated.
        # P and V can be any object that supports indexing, so it is important
        # that you know they define a valid MDP before calling the
        # _bellmanOperator method. Otherwise the results will be meaningless.
        Q = np.empty((self.A, self.S))
        for aa in range(self.A):
            print('target', np.shape(np.ones(np.shape(self.target))-self.target))
            print('V', np.shape(V))
            print('shape target times V',np.shape(self.target + (np.ones(np.shape(self.target))-self.target) * V ))
            Qval = np.dot(self.P[aa],self.target + (np.ones(np.shape(self.target)) - self.target)* V)
            print('target',np.shape(Qval))

            print('shape P', np.shape(self.P[aa]))
            Q[aa] = Qval.reshape(-1)

            #print(Q[aa].reshape(xi.shape, order='F'))

        # Get the policy and value, for now it is being returned but...
        # Which way is better?
        # 1. Return, (policy, value)
        self.V = Q.max(axis=0).reshape((self.S,1))
        pol = Q.argmax(axis=0)


        ugrid = np.meshgrid(*self.urep)
        self.policy_u = np.empty((self.S,len(self.urep)))

        for u_index,u_grid_index in enumerate(ugrid):
            u_row = u_grid_index.flatten()
            for i,u in enumerate(pol):
                #print(i,u)
                self.policy_u[i][u_index] =u_row[u]

        return (Q.argmax(axis=0), Q.max(axis=0))
        # 2. update self.policy and self.V directly
        # self.V = Q.max(axis=1)
        # self.policy = Q.argmax(axis=1)