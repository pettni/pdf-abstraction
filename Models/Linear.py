""" Routines for handeling linear time invariant systems.

author S. Haesaert
"""
import numpy as np
import itertools
import polytope as pc
from numpy import linalg as LA
import matplotlib.pyplot as plt
from ApprxSimulation.LTI_simrel import eps_err

from scipy.stats import norm
from functools import reduce # Valid in Python 2.6+, required in Python 3
import operator
from ApprxSimulation.Visualize import plot_rel, patch_ellips

import control
from Models.MDP import Markov


class LTI:
    """Define a discrete-time linear time invariant system"""

    def __init__(self, a, b, c, d, x=None, bw=None, W=None, u=None):
        self.a = a
        self.b = b
        self.c = c  # currently unused
        self.d = d  # currently unused
        self.dim = len(a)
        self.m = b.shape[1]
        self.X = x
        if W is None:
            if bw is None:
                self.W = None
                self.bw = None
            else:
                self.bw = bw
                self.W = bw.dot(bw.T)
        else:
            self.W = W

        self.U = u

    def setU(self, u=None):
        if isinstance(u,pc.Polytope):
            self.U = u
            return self.U
        if u is None:
            print('Warning no inputspace given')
            if self.U is None:
                print('Define standard box polytope 0-1')
                self.U = pc.box2poly(np.kron(np.ones((self.m, 1)), [-1, 1]))
                return self.U
            else:
                return self.U

    def setX(self, x=None):
        if isinstance(x,pc.Polytope):
            self.X = x
            return self.X
        else:
            print('Warning no state space given')
            if self.X is None:
                print('Define standard box polytope -1,1')
                self.X = pc.box2poly(np.kron(np.ones((self.dim, 1)), np.array([[-1, 1]])))
                return self.X
            else:
                return self.X

    def setBw(self, bw=None):
        if isinstance(bw,np.ndarray) :
            self.bw = bw
            self.W = bw.dot(bw.T)

            return self.bw
        if bw is None:
            print('Warning no matrix BW given')
            if self.bw is None:
                print('Define matrix Bw')
                self.bw = np.eye(self.dim)
                self.W = self.bw.dot(self.bw.T)

                return self.bw
            else:
                print('keep matrix Bw')
                return self.bw

    def abstract(self,d, un=3, verbose = True, Accuracy =True):
        ## Unpack LTI
        d=d.flatten()
        A = self.a
        B = self.b
        C = self.c
        Bw = self.setBw()
        U = self.setU()

        # check that Bw is a diagonal
        # = np.sum(np.absolute(np.dot(Bw,Bw.transpose()))) - np.trace(np.absolute(np.dot(Bw,Bw.transpose())))
        assert np.sum(np.absolute(np.dot(Bw,Bw.transpose()))) - np.trace(np.absolute(np.dot(Bw,Bw.transpose()))) == 0
        vars = np.diag(np.dot(Bw,Bw.transpose()))

        X = self.setX()
        n = self.dim

        rad = LA.norm(d, 2)
        lx, ux = pc.bounding_box(self.X)  # lower and upperbounds over all dimensions
        remainx = np.remainder((ux-lx).flatten(),d.flatten())
        remainx = np.array([d.flatten()[i]-r if r!=0 else 0 for i,r in enumerate(remainx) ]).flatten()
        lx =lx.flatten() - remainx/2
        ux =ux.flatten() + d

        if Accuracy:
            Dist = pc.box2poly(np.diag(d).dot(np.kron(np.ones((self.dim, 1)), np.array([[-1, 1]]))))

            M_min, K_min, eps_min = eps_err(self, Dist)

        if verbose == True and n == 2:
            # plot figure of x
            figure = plt.figure()
            figure.add_subplot('111')
            axes = figure.axes[0]
            axes.axis('equal')  # sets aspect ration to 1

            # compute limits X
            plt.xlim(np.array([lx[0], ux[0]]))
            plt.ylim(np.array([lx[1], ux[1]]))

        # GRID state
        srep = tuple()
        sedge=tuple()
        for i, dval in enumerate(d):
            srep += (np.arange(lx[i], ux[i], dval)+dval/2,)
            sedge += (np.arange(lx[i], ux[i]+dval, dval),)

        grid = np.meshgrid(*srep)
        xn = np.size(grid[0]) # number of finite states

        # grid input
        urep = tuple()
        lu, uu = pc.bounding_box(self.U)  # lower and upperbounds over all dimensions
        for i, low in enumerate(lu):
            urep += (np.linspace(lu[i], uu[i], un, endpoint=True),)

        ugrid = np.meshgrid(*urep)
        un = np.size(ugrid[0])  # number of finite states



        transition = np.zeros((un, xn, xn))
        print(transition.shape)
        transition.fill(-10)
        for u_index, u in enumerate(itertools.product(*urep)):
            P = tuple()
            for s,sstate in enumerate(itertools.product(*srep)):
                mean = np.dot(A,np.array(sstate))+np.dot(B,np.array(u))  # Ax

                # compute probability in each dimension
                Pi = tuple()
                for i in range(n):
                    Pi += (np.diff(norm.cdf(sedge[i], mean[i], vars[i]**.5)),) # probabilities for different dimensions

                # multiply over dimensions
                P += (np.array([[reduce(operator.mul, p, 1) for p in itertools.product(*Pi)]]),)

            prob = np.concatenate(P, axis = 0)
            transition[u_index] = prob







        mdp_grid = Markov(transition, srep, urep, sedge)



        if verbose == True and n == 2:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(grid[0].flatten(), grid[1].flatten(), label='Finite states', color='k', s=10, marker="o")

            plt.xlabel('x')
            plt.ylabel('y')
            if Accuracy:
                patch = patch_ellips((eps_min ** -2) * M_min, pos=None, number=20)
                ax.add_patch(patch)


            # if B is lower dimension than A, give some info on the stationary points.
                if self.m<self.dim:
                    A1=np.eye(2)-self.a
                    AB=np.hstack((A1,self.b))
                    ratio =LA.inv(AB[:,1:]).dot(AB[:,0])
                    stablepoints = ratio[0] *srep[0]
                    ax.scatter(srep[0].flatten(), stablepoints.flatten(), label='Equilibrium states', color='r', s=20, marker="+")
            plt.legend()


                    # plt.tight_layout()


            plt.show()

        return mdp_grid

class POMDP(LTI):
    "All data for sensor based modeling is enclosed in this class"
    def __init__(self, lti, H, V, P, mean):
        """
        create POMDP based on lti system and observation model
        :param lti:  lti model
        :param H: matrix z=H x+v
        :param V: covariance of v ~ N(0,V)
        :param P: covariance of (x-xhat)
        :param mean:  xhat
        """
        LTI.__init__(self, lti.a, lti.b, lti.c, lti.d, x=lti.X, bw=lti.bw, u=lti.U)
        self.H = H
        self.V = V
        self.W = np.dot(self.bw, self.bw.transpose())
        self.mean = mean
        self.P = P

    def predict(self, u, x=None, P=None):
        """
        :param x: mean state estimate x_{k|k}
        :param u: input action u_k
        :param P: state est. covariance P_{k|k}
        :return: (x_{k+1|k}, P_{k+1|k})
        """
        x = self.a.dot(x)+self.b.dot(u)
        P = self.a.dot(P).dot(self.a.T)+self.W

        return x, P

    def update(self, x, P, e=None, z=None):
        """
        :param x: mean state estimate x_{k|k-1}
        :param P: state est. covariance P_{k|k}
        :param e: zk-C_z x_{k|k-1}
        :param z: zk
        :return:(x_{k|k}, P_{k|k})
        """
        S = self.H.dot(P).dot(self.H.transpose())+self.V
        L = P.dot(self.H.transpose()).dot(LA.inv(S))
        P_update = np.dot((np.eye(self.dim) - L.dot(self.H)), P)
        x_update = np.zeros((self.dim, 1))
        if e is None:
            if z is None:
                return x_update, P_update
            else:
                e = z-self.H.dot(x)

        x += L.dot(e)
        return x_update, P_update

    def kalman(self):
        """
        :return: P = prediction variance
        """
        X, eigval, G = control.dare(self.a.T,self.H.T, self.W,self.V)
        L = G.dot(LA.inv(self.a.T)).T

        return L, X

    def beliefmodel(self):
        """
        :return: model of belief
        """
        # create LTI system
        belief_model = beliefmodel(self.a, self.b, self.H, self.mean, self.P, self.V, self.W)
        return belief_model

class beliefmodel:

    def __init__(self, a, b, H, mean, P, V, W):
        self.a = a
        self.b = b
        self.H = H
        self.state = (mean,P) # initial state (k=0,k=0)
        self.V = V
        self.W = W
        self.dim = len(a)

    def kalman(self):
        """
        :return: P = prediction variance
        """
        X, eigval, G = control.dare(self.a.T,self.H.T, self.W,self.V)
        L = G.dot(LA.inv(self.a.T)).T
        return L, X

    def to_LTI(self,c):
        """
        :return: LTI model of belief
        """
        # create LTI system

        L,P = self.kalman()
        S = self.H.dot(P).dot(self.H.transpose())+self.V

        Cov = L.dot(S).dot(L.T)
        self.c=c
        belief_model = LTI(self.a, self.b, self.c, None, W = Cov)

        return belief_model

    def predict(self, u, x=None, P=None):
        """
        :param x: mean state estimate x_{k|k}
        :param u: input action u_k
        :param P: state est. covariance P_{k|k}
        :return: (x_{k+1|k}, P_{k+1|k})
        """
        if x is None:
            x = self.state[0]

        if P is None:
            P = self.state[1]

        x = self.a.dot(x)+self.b.dot(u)
        P = self.a.dot(P).dot(self.a.T)+self.W

        return x, P

    def update(self, x, P, e=None, z=None, simulate=True):
        """
        :param x: mean state estimate x_{k|k-1}
        :param P: state est. covariance P_{k|k}
        :param e: zk-C_z x_{k|k-1}
        :param z: zk
        :return:(x_{k|k}, P_{k|k})
        """
        S = self.H.dot(P).dot(self.H.transpose())+self.V
        L = P.dot(self.H.transpose()).dot(LA.inv(S))
        P_update = np.dot((np.eye(self.dim) - L.dot(self.H)), P)
        x_update = np.zeros((self.dim, 1))
        if e is None:
            if z is None:
                if simulate:
                    Cov = S.tolist()
                    mean = np.zeros((self.H.shape[0], 1)).T.flatten().tolist()
                    e = np.random.multivariate_normal(mean, Cov, 1).T
                else:
                    return x_update, P_update
            else:
                e = z-self.H.dot(x)

        x_update = x + L.dot(e)
        return x_update, P_update


    def simulate(self, u, x=None, P=None):
        """
        :param x: mean state estimate x_{k|k-1}
        :param P: state est. covariance P_{k|k}
        :param e: zk-C_z x_{k|k-1}
        :param z: zk
        :return:(x_{k|k}, P_{k|k})
        """

        # currently this only iterates the P matrix, not the state (due to the stochasticity)
        if x is None:
            x = self.state[0]

        if P is None:
            P = self.state[1]

        (x_predict, P_predict) = self.predict(u, x=x, P=P)

        (x_update, P_update) = self.update(x_predict, P_predict,simulate=True)

        self.state = (x_update, P_update)

        return x_update, P_update

