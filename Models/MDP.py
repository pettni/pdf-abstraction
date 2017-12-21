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
from scipy.sparse import vstack,hstack

from mdptoolbox.mdp import *
import numpy as np
import polytope as pc
import itertools
import numpy.linalg as LA

class Markov(MDP):
    """Simple Markov Decision process

      Markov decision process for reaching target set.

    """

    def __init__(self,transitions, srep, urep,sedge, M=None, K=None, eps=None,T2x = None):



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
        self.act_inputs = None
        self.dfa = None
        self.final = None
        self.T2x = T2x

        self.trans_qs = None
        # accuracy
        self.M = M
        self.eps = eps
        self.K = K
        self.s_finite = None

        self.output_cst = dict([(s, sstate) for s, sstate in enumerate(itertools.product(*srep))])
        self.input_cst = dict([(u, uvalue) for u, uvalue in enumerate(itertools.product(*urep))])


    def state_fnc(self, s):
        return self.output_cst[s]

    # def input_fnc(self,uvalue):
    #     uval_discrete = tuple()
    #     print(self.input_cst)
    #     for ui,uval in enumerate(uvalue.flatten()):
    #         uval_discrete += (min(self.urep[ui].flatten(), key=lambda x: abs(x - uval)),)
    #     return self.input_cst[uval_discrete]

    def nextsets(self, s_next):
        if self.s_finite is None:
            self.s_finite = np.array(list(itertools.product(*self.srep)))  # compute the grid points

        if self.M is None:
            print("WARNING no M matrix given")
            self.M =np.eye(len(self.srep))

        if self.eps is None:
            print("WARNING no epsilon give")
            self.eps = 1

        # quantify the weighted difference between s_next, and values of s
        sdiff = self.s_finite-np.ones((self.s_finite.shape[0],1)).dot(s_next.reshape((1,-1)))
        error=np.diag(sdiff.dot(self.M).dot(sdiff.T))
        s_range = np.arange(self.S-1) # minus to remove dummy variable
        return s_range[error<=self.eps**2], self.s_finite[error<=self.eps**2]

    def settarget(self, Target=None):
        if Target is None:
             return self.target

        if isinstance(Target,pc.Polytope):
            self.target= np.array([[1.] if np.array(s) in Target else [0.] for s in itertools.product(*self.srep)])
            print(np.shape(self.target))
        else:
            self.target = Target

    def map_dfa_inputs(self, dictio, regions):
        """

        :param dict: dictionary with keys input numbers and values sets of ap, A =number of inputs
        :param regions: dictionary with keys = ap, and values polytopes
        :return: matrix with size A x states
        """
        act_inputs = np.zeros((len(dictio.keys()),self.P[0].shape[0]-1))
        in_regions = dict()
        nin_regions = dict()

        if (self.eps is None) | (self.eps ==0):
            for input_i in regions.keys():
                in_regions[input_i] = np.array([[1.] if self.T2x.dot(np.array(s)) in regions[input_i] else [0.] for s in itertools.product(*self.srep)])

            for input_i in regions.keys():
                nin_regions[input_i] = np.ones(in_regions[input_i].shape)-np.array([[1.] if self.T2x.dot(np.array(s)) in regions[input_i] else [0.] for s in itertools.product(*self.srep)])

        else :
            u, s, v = LA.svd(self.M)
            Minvhalf = LA.inv(v).dot(np.diag(np.power(s, -.5)))
            Minhalf = np.diag(np.power(s, .5)).dot(v)
            # eps is not =0,
            for input_i in regions.keys(): # for each region, which is a polytope. Check whether it can be in it
                #Big_polytope = regions[input_i] #<--- decrease size polytope

                A = regions[input_i].A.dot(self.T2x).dot(Minvhalf)
                b = regions[input_i].b

                scaling = np.zeros((A.shape[0],A.shape[0]))
                for index in range(A.shape[0]):
                    scaling[index,index] = LA.norm(A[index,:])**-1
                print('check norm of rows', scaling.dot(A))
                A = scaling.dot(A).dot(Minhalf)

                b = scaling.dot(b) + self.eps

                assert regions[input_i].A.shape == A.shape
                assert regions[input_i].b.shape == b.shape

                in_regions[input_i] = np.array(
                    [[1.] if np.all(A.dot(np.array(s))-b<=0) else [0.] for s in
                     itertools.product(*self.srep)])

            for input_i in regions.keys():
                # quantify whether a state could be outside the polytope
                A = regions[input_i].A.dot(self.T2x).dot(Minvhalf)
                b = regions[input_i].b

                scaling = np.zeros((A.shape[0],A.shape[0]))
                for index in range(A.shape[0]):
                    scaling[index,index] = LA.norm(A[index,:])**-1

                A = scaling.dot(A).dot(Minhalf)
                b = scaling.dot(b) - self.eps




                nin_regions[input_i] = np.ones(in_regions[input_i].shape) - np.array(
                    [[1.] if np.all(A.dot(np.array(s))-b<=0)else [0.] for s in
                     itertools.product(*self.srep)])

        for aps in dictio.keys() :

            if dictio[aps] :
                set_ap = set(dictio[aps])
            else :
                set_ap = set()

            act_inputs[aps] = np.prod(np.block([[in_regions[input_i] if input_i in set_ap else nin_regions[input_i] for input_i in regions.keys()]]),axis=1)
            # ap of interest are now given as
        assert act_inputs.sum(axis=0).min()>0
        #fill in with zeros for the dummy state

        act_inputs = np.block([[act_inputs,np.zeros((len(dictio.keys()),1))]])
        print(act_inputs)


        self.act_inputs = act_inputs



        return act_inputs

    def setdfa(self,dfa,final):
        self.dfa = dfa
        self.final =final

    def reach_dfa(self,V = None, recursions=1, delta=0 ):
        # TODO : Add delta
        assert self.act_inputs is not None
        assert self.dfa is not None
        assert self.final is not None
        if V is None:
            # this V should be a reference to the data rather than a copy
            V = np.zeros((self.dfa.N, self.S))

        assert V.shape in ((self.dfa.N, self.S),)

        # let q\in Q be states of DFA
        # let S \in SS be states of MDP

        # create indicator functions for final, Q\(final)

        accept = lambda q: map(lambda qi: int(qi in self.final), q)

        naccept= lambda q: map(lambda qi: 1-int(qi in self.final), q)
        Accept= np.kron(np.array(accept(range(self.dfa.N))).reshape((self.dfa.N,1)), np.ones((1,self.S)))
        nAccept = np.diag(np.array(naccept(range(self.dfa.N))))

        Tnew = hstack(self.dfa.Tmat).toarray()
        #print(Tnew.toarray())
        if self.trans_qs is None:
            trans_qqa = np.zeros((self.dfa.N,self.dfa.N,len(self.dfa.Tmat))) # q,q',act
            trans_qs = np.zeros((self.dfa.N,self.dfa.N,self.S)) # q, q',S'

            for q in range(self.dfa.N):
                trans_qqa[q] = Tnew[q,:].reshape((self.dfa.N, -1), order= "F")
                array = np.zeros((self.dfa.N,self.S))
                bool_array =(trans_qqa[q].dot(self.act_inputs) < 1)
                array[bool_array] = 1000.
                trans_qs[q] = array #np.array((array < 1), dtype=np._float)*1000.0 # penalise impossible transitions

            trans_qs[:,:,-1]=np.zeros((self.dfa.N,self.dfa.N)) # set dummy state equal to zero
            self.trans_qs = trans_qs

        # [T[0] T[1] T[2] ... ]
        # given q, S
        # next SxAct -> prob(S')
        # W = 1accept(qnext) + 1_{not accept }(qnext)  V
        pol = np.zeros((self.dfa.N, self.S))
        V_new = np.zeros((self.dfa.N, self.S))
        for rec in range(recursions):

            for q in range(self.dfa.N):
                W = np.amin(Accept+ nAccept.dot(V)+self.trans_qs[q],axis =0 ) # 1 x S'
                W_a = np.block([[W.dot(self.P[a].T)] for a in range(self.A)])
                if rec == recursions-1 : # at last step also comput the policy
                    pol[q] = W_a.argmax(axis = 0)
                if delta == 0:
                    V_new[q] = W_a.max(axis = 0)   #max_{s_action}[ s_action X S]
                else:
                    V_new[q] = W_a.max(axis = 0) - delta  #max_{s_action}[ s_action X S]

                V_new[q] = np.clip(V_new[q],0,1) # limit to values between zero and one
                #print(q, V_new[q].sum(axis =0))
            V = V_new



        W = np.amin(Accept+ nAccept.dot(V)+self.trans_qs[self.dfa.init[0]],axis =0 )
        return V,pol, W

    def reach_bell(self, V = None):
        assert False
        # old implementation
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

    def policy_dfa(V):
        pass
    # create steady state policy.




class Rpol(): # refined policy

    def __init__(self, MDP, V, W,policy):
        self.dfa = MDP.dfa
        self.state = self.dfa.init[0]  ## composed of discrete state only
        self.trans = MDP.dfa.Tmat  # a qq' discrete transitions

        self.V = V
        self.W = W
        self.MDP= MDP # abstract states in MDP
        self.aps = None  # aps mapping (S,aps)
        self.pol = policy
        self.s_finite= None
        self.input_ap = None
        self.ap_regions = None

        # interface
        self.K = np.zeros((len(MDP.urep),len(MDP.srep)))

    def __call__(self, s_concrete, transformed = True):
        """
        :param s_next:  s_{k+1}
        :return: input
        """

        if s_concrete.shape[1]>1:
            u = np.zeros((len(self.MDP.urep),s_concrete.shape[1] ))
            for i in range(s_concrete.shape[1]):
                u[:,i] = self.__call__( s_concrete[:,[i]],transformed = transformed).flatten()
            return u
        if not transformed:
            s_next = LA.inv(self.MDP.T2x).dot(s_concrete)
        else:
            s_next = s_concrete

        # get next discrete state
        self.state = self.nextq(self.state, s_next)

        # get next abstract MDP state
        s_abstract, s_abstract_v = self.nexts(self.state, s_next)

        # compute abstract input
        if s_abstract < self.MDP.S - 1:
            u_ab = np.array(self.MDP.input_cst[self.pol[self.state, s_abstract]]).reshape(-1, 1)
        else:
            u_ab = np.zeros((len(self.MDP.input_cst[0]),1))



        # refine

        u = self.interface(u_ab, s_abstract, s_next)

        return u # input


    def cst(self, s_concrete, transformed = True):
        """
        :param s_next:  s_{k+1}
        :return: input
        """

        if s_concrete.shape[1]>1:
            u = np.zeros((len(self.MDP.urep),s_concrete.shape[1] ))
            for i in range(s_concrete.shape[1]):
                u[:,i] = self.cst(s_concrete[:,i].reshape((-1,1)),transformed = transformed).flatten()
            return u
        if not transformed:
            s_next = LA.inv(self.MDP.T2x).dot(s_concrete)
        else:
            s_next = s_concrete


        # get next discrete state
        q = self.nextq( self.dfa.init[0], s_next)

        # get next abstract MDP state
        s_abstract, s_abstract_v = self.nexts(q, s_next)

        # compute abstract input
        if s_abstract < self.MDP.S - 1:
            u_ab = np.array(self.MDP.input_cst[self.pol[q, s_abstract]]).reshape(-1, 1)
        else:
            u_ab = np.zeros((len(self.MDP.input_cst[0]),1))


        # refine

        u = self.interface(u_ab, s_abstract, s_next)

        return u # input

    def interface(self,uab,sab,s):
        """

        :param uab:
        :param sab:
        :param s: after transform concrete state to normalised
        :return:
        """
        # only works after transform
        u = self.K.dot(s-sab) + uab
        return u

    def nexts(self, q_next, s_next):
        """
        :param q_next: next state in DFA
        :param s_next: next concrete state (after transform)
        :return: next abstrac state
        """

        indexes, values = self.MDP.nextsets(s_next)

        # pick argument of max value
        try:
            s_abstract = indexes[self.V[q_next][indexes].argmax()]
            s_abstract_v = values[self.V[q_next][indexes].argmax()]
        except ValueError:
            s_abstract = self.MDP.S
            s_abstract_v = 0

        return s_abstract, s_abstract_v

    def val_concrete(self,s_concrete):
        """
        :param s_concrete: before transformation state
        :return:
        """
        if s_concrete.shape[1] > 1:
            val = np.zeros((s_concrete.shape[1],))
            for i in range(s_concrete.shape[1]):
                valdd = self.val_concrete(s_concrete[:, i].reshape((-1, 1))).flatten()

                try :
                    val[i] = valdd
                except :
                    val[i] = 0
                    print('ERR')

            return val
        s_next = LA.inv(self.MDP.T2x).dot(s_concrete) # transform to normalised

        # get next discrete state
        self.state = self.nextq(self.state, s_next)

        # get next abstract MDP state
        s_abstract, s_abstract_v = self.nexts(self.state, s_next)

        if s_abstract < self.MDP.S-1:
            val = self.W[s_abstract]
        else:
            val = 0

        return np.array([[val]])

    def set_regions(self, dictio, regions):
        self.input_ap = dict([(dictio[i],i) for i in dictio.keys()])
        self.ap_regions = regions

    def nextq(self, q, s):
        """
        :param q: current q
        :param s: next s after transform to normalised
        :return: next q
        """
        aps = tuple()
        for input_i in self.ap_regions.keys():
            if np.all((self.ap_regions[input_i].A.dot(self.MDP.T2x).dot(np.array(s).reshape((-1,1)))- self.ap_regions[input_i].b.reshape((-1,1))) < 0) :
                aps += (input_i,)
        # => next input!!!
        i = self.input_ap[aps]


        return self.dfa.T(i)[q].argmax()
