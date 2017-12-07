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

"""Routines for the quantification of the simulation relations"""
import polytope as pc
import cvxpy as cvx
import numpy as np
import cvxopt


def eps_err(lti,Dist,lamb=.9):
    """
    Quantify accuracy of simulation with respect to disturbance given as a polytope
    :param lti: contains dynamics matrix lti.a, lti.b
    :param Dist: The disturbance given as a polytope
    :return: Invariant set R and epsilon
    """
    n = lti.dim
    m = lti.m
    A = lti.a
    B = lti.b
    C = lti.c

    Vertices = pc.extreme(Dist)
    print(Vertices)
    # define variables
    Minv = cvx.Semidef(n)
    L = cvx.Variable(m,n)
    eps2 = cvx.Semidef(1)
    lam = cvx.Parameter(sign="positive",value = lamb)
    basic = cvx.bmat([[cvx.diag(np.ones((n,1))*lam) * Minv, np.zeros((n,1)),  Minv * A.T + L.T * B.T ],
                      [np.zeros((1,n)), (1-lam) * eps2, np.zeros((1,n))],
                      [A * Minv + B * L , np.zeros((n,1)), Minv]])

    constraintstup = tuple()
    ri =  np.zeros((n,1))
    for i in range(Vertices.shape[0]):
        ri = Vertices[i].reshape((n,1))

        print(ri)
        rmat = cvx.bmat([[np.zeros((n, n)), np.zeros((n, 1)), np.zeros((n, n))],
                         [np.zeros((1, n)), np.zeros((1, 1)), ri.T],
                         [np.zeros((n, n)), ri, np.zeros((n, n))] ]   )
        constraintstup += (basic + rmat,)
    constraints = list(constraintstup)



    obj = cvx.Minimize(eps2)
    prob = cvx.Problem(obj, constraints)


    prob.solve()  # Returns the optimal value.
    print "status:", prob.status
    print "optimal value", prob.value
    print "optimal var", Minv.value, L.value

    return  prob.value
