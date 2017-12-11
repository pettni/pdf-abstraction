# Copyright (c) 2013-2017 by California Institute of Technology
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the California Institute of Technology nor
#    the names of its contributors may be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL CALTECH
# OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF
# USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
# OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
# SUCH DAMAGE.
"""Markov processes"""

from mdptoolbox.mdp import *
import numpy as np
import polytope as pc
import itertools

class Markov(MDP):
    """Simple Markov Decision process

      Markov decision process for reaching target set.

    """

    def __init__(self,transitions, srep, urep,sedge):



        S =transitions.shape[1]
        A = transitions.shape[0]
        reward = np.ones( (S,A))
        print(reward.shape, transitions.shape)
        discount = None
        epsilon = None
        max_iter = None

        MDP.__init__(self, transitions, reward, discount, epsilon, max_iter,skip_check=True)
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

        Q = np.empty((self.A, self.S))
        for aa in range(self.A):
            qval = np.dot(self.P[aa],self.target + (np.ones(np.shape(self.target)) - self.target)* V)
            Q[aa] = qval.reshape(-1)

        self.V = Q.max(axis=0).reshape((self.S,1))
        pol = Q.argmax(axis=0)

        ugrid = np.meshgrid(*self.urep)
        self.policy_u = np.empty((self.S,len(self.urep)))

        for u_index,u_grid_index in enumerate(ugrid):
            u_row = u_grid_index.flatten()
            for i,u in enumerate(pol):
                self.policy_u[i][u_index] =u_row[u]

        return (Q.argmax(axis=0), Q.max(axis=0))