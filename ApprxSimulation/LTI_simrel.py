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
import scipy.optimize
import itertools

def eps_err(lti,Dist,lamb=.99999):
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


    def f_opt(val):
        lam.value = val
        try:
            prob.solve()
        except cvx.error.SolverError :
            return np.inf

        return eps2.value**.5

    lam_opt = scipy.optimize.fminbound(lambda val: f_opt(val), 0,1)
    lam.value = lam_opt
    prob.solve()
    eps_min = eps2.value ** .5
    M_min = inv(Minv.value)
    K_min = L.value*Minv.value

    print "status:", prob.status
    print "optimal epsilon", eps_min
    print "optimal M", M_min
    print "Optimal K", K_min


    return M_min, K_min, eps_min






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
                pass
                #print('cvx.error.SolverError')
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
                pass #print('cvx.error.SolverError')
            # Use expr.value to get the numerical value of
            # an expression in the problem.
            if prob.status == cvx.OPTIMAL:
                if eps2.value ** .5 < eps_min:
                    eps_min = eps2.value ** .5
        eps_values.append(eps_min)
        cost = d_val[0] ** -1 * d_val[1] ** -1 * eps_min ** 2
        if cost[0] <= optval:
            optval = cost
            d_opt= d_val

    # Plot entries of x vs. gamma.
    plt.subplot(212)
    plt.plot([dval[1] for dval in d_vals], [xi for xi in eps_values],label = 'Epsilon')
    plt.tight_layout()

    plt.plot([dval[1] for dval in d_vals],[dval[0]**-1 * dval[1]**-1 *eps_values[i]**2 for dval in d_vals], label = 'Cost gridding a square')


    plt.xlabel(r'd[1]', fontsize=16)
    plt.ylabel(r'epsilon', fontsize=16)
    plt.title(r' Tune grid ratio ', fontsize=16)
    #plt.yscale('log')

    plt.show()

    return d_opt, d_vals,eps_values




def evaluateR(M,r):
    return r.T*M*r


def eps_err_dist(lti,grid,dist,lamb=.99999):
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
    Vertices_grid = pc.extreme(grid)
    if type(lti.T2x) is np.ndarray:
        Apol = dist.A.dot(lti.T2x)
        dist = pc.Polytope(A=Apol, b=dist.b)
        Vertices_dist = pc.extreme(dist)
    else:
        Vertices_dist = pc.extreme(dist)



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
    for i,j in itertools.product(range(Vertices_grid.shape[0]),range(Vertices_dist.shape[0])):
        ri = Vertices_grid[i].reshape((n,1))
        rj = Vertices_dist[i].reshape((n, 1))
        rmat = cvx.bmat([[np.zeros((n, n)), np.zeros((n, 1)), np.zeros((n, n))],
                         [np.zeros((1, n)), np.zeros((1, 1)), ri.T+rj.T],
                         [np.zeros((n, n)), ri+rj, np.zeros((n, n))] ]   )
        constraintstup += (basic + rmat >> 0,)


    constraints = list(constraintstup)



    obj = cvx.Minimize(eps2)
    prob = cvx.Problem(obj, constraints)


    def f_opt(val):
        lam.value = val
        try:
            prob.solve()
        except cvx.error.SolverError :
            return np.inf

        return eps2.value**.5

    lam_opt = scipy.optimize.fminbound(lambda val: f_opt(val), 0,1)
    lam.value = lam_opt
    prob.solve()
    eps_min = eps2.value ** .5
    M_min = inv(Minv.value)
    K_min = L.value*Minv.value

    print "status:", prob.status
    print "optimal epsilon", eps_min
    print "optimal M", M_min
    print "Optimal K", K_min


    return M_min, K_min, eps_min



def eps_err_dist_tuned(lti,grid,dist,lamb=.99999):
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
    Vertices_grid = pc.extreme(grid)
    if type(lti.T2x) is np.ndarray:
        Apol = dist.A.dot(lti.T2x)
        dist = pc.Polytope(A=Apol, b=dist.b)
        Vertices_dist = pc.extreme(dist)
    else:
        Vertices_dist = pc.extreme(dist)

     # define variables
    Minv = cvx.Semidef(n)
    L = cvx.Variable(m,n)
    d = cvx.Parameter(2, 1)
    eps2 = cvx.Semidef(1)
    lam = cvx.Parameter(sign="positive",value = lamb)
    basic = cvx.bmat([[cvx.diag(np.ones((n,1))*lam) * Minv, np.zeros((n,1)),  Minv * A.T + L.T * B.T ],
                      [np.zeros((1,n)), (1-lam) * eps2, np.zeros((1,n))],
                      [A * Minv + B * L , np.zeros((n,1)), Minv]])

    cmat = cvx.bmat([[Minv, Minv * C.T],[C* Minv, np.eye(C.shape[0])]])
    constraintstup = (cmat >> 0,)

    ri =  np.zeros((n,1))
    for i,j in itertools.product(range(Vertices_grid.shape[0]),range(Vertices_dist.shape[0])):
        ri = Vertices_grid[i].reshape((n,1))
        rj = Vertices_dist[i].reshape((n, 1))
        rmat = cvx.bmat([[np.zeros((n, n)), np.zeros((n, 1)), np.zeros((n, n))],
                         [np.zeros((1, n)), np.zeros((1, 1)), ri.T*cvx.diag(d)+rj.T],
                         [np.zeros((n, n)), cvx.diag(d)*ri+rj, np.zeros((n, n))] ]   )
        constraintstup += (basic + rmat >> 0,)


    constraints = list(constraintstup)


    obj = cvx.Minimize(eps2)
    prob = cvx.Problem(obj, constraints)


    def f_opt(val):
        lam.value = val
        try:
            prob.solve()
        except cvx.error.SolverError :
            return np.inf
        return eps2.value**.5



    def f_optd(val):
        d_val = np.array([[math.cos(val)], [math.sin(val)]])
        d.value=d_val
        lam_opt = scipy.optimize.fminbound(lambda val: f_opt(val), 0, 1, maxfun=10)
        lam.value = lam_opt
        prob.solve()
        return d_val[0] ** -1 * d_val[1] ** -1 * eps2.value

    vald = scipy.optimize.fminbound(lambda val: f_optd(val), 0,math.pi/2,maxfun =40)
    d_val = np.array([[math.cos(vald[0])], [math.sin(vald[0])]])
    d.value = d_val

    lam_opt = scipy.optimize.fminbound(lambda val: f_opt(val), 0,1,maxfun =10)
    lam.value = lam_opt
    prob.solve()


    eps_min = eps2.value ** .5
    M_min = inv(Minv.value)
    K_min = L.value*Minv.value

    print "status:", prob.status
    print "optimal epsilon", eps_min
    print "optimal M", M_min
    print "Optimal K", K_min


    return d_val,M_min, K_min, eps_min