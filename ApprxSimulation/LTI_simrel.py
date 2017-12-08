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
from numpy.linalg import inv
import matplotlib.pyplot as plt
import math

def eps_err(lti,Dist,lamb=.9, verbose =True):
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
     # define variables
    Minv = cvx.Semidef(n)
    L = cvx.Variable(m,n)
    eps2 = cvx.Semidef(1)
    lam = cvx.Parameter(sign="positive",value = lamb)
    basic = cvx.bmat([[cvx.diag(np.ones((n,1))*lam) * Minv, np.zeros((n,1)),  Minv * A.T + L.T * B.T ],
                      [np.zeros((1,n)), (1-lam) * eps2, np.zeros((1,n))],
                      [A * Minv + B * L , np.zeros((n,1)), Minv]])

    cmat = cvx.bmat([[Minv, Minv * C.T],[C* Minv, np.eye(C.shape[0])]])
    constraintstup = (cmat >> 0,)

    ri =  np.zeros((n,1))
    for i in range(Vertices.shape[0]):
        ri = Vertices[i].reshape((n,1))
        rmat = cvx.bmat([[np.zeros((n, n)), np.zeros((n, 1)), np.zeros((n, n))],
                         [np.zeros((1, n)), np.zeros((1, 1)), ri.T],
                         [np.zeros((n, n)), ri, np.zeros((n, n))] ]   )
        constraintstup += (basic + rmat >> 0,)
    constraints = list(constraintstup)



    obj = cvx.Minimize(eps2)
    prob = cvx.Problem(obj, constraints)

    lam_vals = np.logspace(-.01,-2) # values to try
    eps_values = []         # values for which there is a solution
    lam_values = []         # values for which there is a solution
    eps_min = []            # track minimum value
    M_min = []
    K_min = []
    for val in lam_vals:
        lam.value = val
        try:
            prob.solve()
        except cvx.error.SolverError :
            print('cvx.error.SolverError')
        # Use expr.value to get the numerical value of
        # an expression in the problem.
        if prob.status == cvx.OPTIMAL:

            eps_values.append(eps2.value ** .5)
            lam_values.append(lam.value)
            if eps2.value ** .5 < eps_min:
                eps_min = eps2.value ** .5
                M_min = inv(Minv.value)
                K_min = L.value*Minv.value

    print "status:", prob.status
    print "optimal value", eps_min
    print "optimal var", M_min, K_min

    print "optimal var", np.array([ [evaluateR(M_min,Vertices[i].reshape((n,1)))]  for i in range(Vertices.shape[0]) ])
    if verbose:
        # Plot entries of x vs. gamma.
        plt.subplot(212)
        plt.plot(lam_values, [xi for xi in eps_values])


        plt.xlabel(r'\lambda', fontsize=16)
        plt.ylabel(r'\epsilon_{i}', fontsize=16)
        plt.xscale('log')
        plt.title(r' Entries of \epsilon vs. \lambda', fontsize=16)

        plt.tight_layout()
        plt.show()

    return  M_min, K_min, eps_min






def tune_d(lti):
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

    Dist = pc.box2poly(np.kron(np.ones((lti.dim, 1)), np.array([[-1, 1]])))

    Vertices = pc.extreme(Dist)
    d=cvx.Parameter(sign="positive",value = 1)
     # define variables
    Minv = cvx.Semidef(n)
    L = cvx.Variable(m,n)
    eps2 = cvx.Semidef(1)
    lam = cvx.Parameter(sign="positive")
    basic = cvx.bmat([[cvx.diag(np.ones((n,1))*lam) * Minv, np.zeros((n,1)),  Minv * A.T + L.T * B.T ],
                      [np.zeros((1,n)), (1-lam) * eps2, np.zeros((1,n))],
                      [A * Minv + B * L , np.zeros((n,1)), Minv]])

    cmat = cvx.bmat([[Minv, Minv * C.T],[C* Minv, np.eye(C.shape[0])]])
    constraintstup = (cmat >> 0,)

    ri =  np.zeros((n,1))
    for i in range(Vertices.shape[0]):
        ri = Vertices[i].reshape((n,1))
        rmat = cvx.bmat([[np.zeros((n, n)), np.zeros((n, 1)), np.zeros((n, n))],
                         [np.zeros((1, n)), np.zeros((1, 1)), ri.T * cvx.diag(np.ones((n,1))*d) ],
                         [np.zeros((n, n)), cvx.diag(np.ones((n,1))*d)*ri, np.zeros((n, n))] ]   )
        constraintstup += (basic + rmat >> 0,)
    constraints = list(constraintstup)



    obj = cvx.Minimize(eps2)
    prob = cvx.Problem(obj, constraints)

    lam_vals = np.logspace(-.01,-2) # values to try
    eps_values = []         # values for which there is a solution
    eps_min = []            # track minimum value
    M_min = []
    K_min = []

    d_vals = np.linspace(0.1,2,10) # values to try
    for d_val in d_vals:
        eps_min =[]
        d.value = d_val
        for val in lam_vals:
            lam.value = val
            try:
                prob.solve()
            except cvx.error.SolverError :
                print('cvx.error.SolverError')
            # Use expr.value to get the numerical value of
            # an expression in the problem.
            if prob.status == cvx.OPTIMAL:
                if eps2.value ** .5 < eps_min:
                    eps_min = eps2.value ** .5
        eps_values.append(eps_min)

    print "status:", prob.status

    # Plot entries of x vs. gamma.
    plt.subplot(212)
    plt.plot(d_vals, [xi for xi in eps_values])


    plt.xlabel(r'd', fontsize=16)
    plt.ylabel(r'\epsilon', fontsize=16)
    plt.title(r' Entries of \epsilon vs. gridsize', fontsize=16)

    plt.tight_layout()
    plt.show()

    return  d_vals,eps_values





def tune_dratio(lti):
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

    Dist = pc.box2poly(np.kron(np.ones((lti.dim, 1)), np.array([[-1, 1]])))

    Vertices = pc.extreme(Dist)
    d=cvx.Parameter(2,1)
     # define variables
    Minv = cvx.Semidef(n)
    L = cvx.Variable(m,n)
    eps2 = cvx.Semidef(1)
    lam = cvx.Parameter(sign="positive")
    basic = cvx.bmat([[cvx.diag(np.ones((n,1))*lam) * Minv, np.zeros((n,1)),  Minv * A.T + L.T * B.T ],
                      [np.zeros((1,n)), (1-lam) * eps2, np.zeros((1,n))],
                      [A * Minv + B * L , np.zeros((n,1)), Minv]])

    cmat = cvx.bmat([[Minv, Minv * C.T],[C* Minv, np.eye(C.shape[0])]])
    constraintstup = (cmat >> 0,)

    ri =  np.zeros((n,1))
    for i in range(Vertices.shape[0]):
        ri = Vertices[i].reshape((n,1))
        rmat = cvx.bmat([[np.zeros((n, n)), np.zeros((n, 1)), np.zeros((n, n))],
                         [np.zeros((1, n)), np.zeros((1, 1)), ri.T * cvx.diag(d) ],
                         [np.zeros((n, n)), cvx.diag(d)*ri, np.zeros((n, n))] ]   )
        constraintstup += (basic + rmat >> 0,)
    constraints = list(constraintstup)



    obj = cvx.Minimize(eps2)
    prob = cvx.Problem(obj, constraints)

    lam_vals = np.logspace(-.01,-2) # values to try
    eps_values = []         # values for which there is a solution
    eps_min = []            # track minimum value
    M_min = []
    K_min = []
    optval = np.array([np.inf])
    d_opt =[]
    d_vals = [] # values to try
    for alpha in np.linspace(0.01* math.pi, 0.4* math.pi, 20) :
        eps_min =[]
        d_val = np.array([[math.cos(alpha)],[math.sin(alpha)]])
        d_vals.append(d_val)
        d.value = d_val
        for val in lam_vals:
            lam.value = val
            try:
                prob.solve()
            except cvx.error.SolverError :
                print('cvx.error.SolverError')
            # Use expr.value to get the numerical value of
            # an expression in the problem.
            if prob.status == cvx.OPTIMAL:
                if eps2.value ** .5 < eps_min:
                    eps_min = eps2.value ** .5
        eps_values.append(eps_min)
        cost = d_val[0] ** -1 * d_val[1] ** -1 * eps_min ** 2
        print(cost,optval)
        if cost[0] <= optval:
            optval = cost
            d_opt= d_val

    print "status:", prob.status

    # Plot entries of x vs. gamma.
    plt.subplot(212)
    plt.plot([dval[1] for dval in d_vals], [xi for xi in eps_values])
    plt.tight_layout()

    plt.plot([dval[1] for dval in d_vals],[dval[0]**-1 * dval[1]**-1 *eps_values[i]**2 for dval in d_vals])


    plt.xlabel(r'd', fontsize=16)
    plt.ylabel(r'\epsilon', fontsize=16)
    plt.title(r' Entries of \epsilon vs. gridsize', fontsize=16)
    plt.yscale('log')

    plt.show()

    return d_opt, d_vals,eps_values




def evaluateR(M,r):
    return r.T*M*r
