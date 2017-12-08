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

from Models.MDP import Markov


class LTI:
    """Define a discrete-time linear time invariant system"""

    def __init__(self, a, b, c, d, x=None, bw=None, u=None):
        self.a = a
        self.b = b
        self.c = c # currently unused
        self.d = d # currently unused
        self.dim = len(a)
        self.m = b.shape[1]
        self.X = x
        self.bw=bw
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
            return self.bw
        if bw is None:
            print('Warning no matrix BW given')
            if self.bw is None:
                print('Define matrix Bw')
                self.bw = np.eye(self.dim)
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

        U = self.U
        X = self.setX()

        n = self.dim
        rad = LA.norm(d, 2)
        lx, ux = pc.bounding_box(self.X)  # lower and upperbounds over all dimensions
        remainx=np.remainder((ux-lx).flatten(),d.flatten())
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






        print('WARNING: UNVERIFIED implementation')

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



