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



        self.output_cst = dict([(s, sstate) for s, sstate in enumerate(itertools.product(*srep))])
        self.input_cst = dict([(uvalue, u) for u, uvalue in enumerate(itertools.product(*urep))])

    def state_fnc(self, s):
        return self.output_cst[s]

    def input_fnc(self,uvalue):
        uval_discrete = tuple()
        print(self.input_cst)
        for ui,uval in enumerate(uvalue.flatten()):
            uval_discrete += (min(self.urep[ui].flatten(), key=lambda x: abs(x - uval)),)
        return self.input_cst[uval_discrete]


    def settarget(self, Target=None):
        if Target is None:
             return self.target

        if isinstance(Target,pc.Polytope):
            self.target= np.array([[1.] if pc.is_inside(Target, np.array(s)) else [0.] for s in itertools.product(*self.srep)])
            print(np.shape(self.target))
        else:
            self.target = Target

    def map_dfa_inputs(self,dictio,regions):
        """

        :param dict: dictionary with keys input numbers and values sets of ap, A =number of inputs
        :param regions: dictionary with keys = ap, and values polytopes
        :return: matrix with size A x states
        """
        act_inputs = np.zeros((len(dictio.keys()),self.P[0].shape[0]-1))
        in_regions = dict()
        nin_regions = dict()

        for input_i in regions.keys():
            in_regions[input_i] = np.array([[1.] if pc.is_inside(regions[input_i], np.array(s)) else [0.] for s in itertools.product(*self.srep)])

        for input_i in regions.keys():
            nin_regions[input_i] = np.ones(in_regions[input_i].shape)-np.array([[1.] if pc.is_inside(regions[input_i], np.array(s)) else [0.] for s in itertools.product(*self.srep)])

        for aps in dictio.keys() :

            if dictio[aps] :
                set_ap = set(dictio[aps])
            else :
                set_ap = set()

            act_inputs[aps] = np.prod(np.block([[in_regions[input_i] if input_i in set_ap else nin_regions[input_i] for input_i in regions.keys()]]),axis=1)
            # ap of interest are now given as
        #fill in with zeros for the dummy state
        act_inputs = np.block([[act_inputs,np.zeros((len(dictio.keys()),1))]])
        return act_inputs






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