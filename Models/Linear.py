""" Routines for handeling linear time invariant systems.

author S. Haesaert
"""
import numpy as np
import itertools
import polytope as pc
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.stats import norm
from functools import reduce # Valid in Python 2.6+, required in Python 3
import operator

from Models.MDP import Markov


class LTI:
    """Define a discrete-time linear time invariant system"""

    def __init__(self, a, b, c, d, x=None, bw=None, u=None):
        print('WARNING: DUMMY implementation')

        self.a = a
        self.b = b
        self.c = c
        self.d = d
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
            print('Warning no state space given')
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

    def abstract(self,d, un=3, verbose = True):
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
        ## split gridding diameter into the dimensions

        n = self.dim
        rad = LA.norm(d, 2)
        lx, ux = pc.bounding_box(self.X)  # lower and upperbounds over all dimensions
        remainx=np.remainder((ux-lx).flatten(),d.flatten())
        remainx = np.array([d.flatten()[i]-r if r!=0 else 0 for i,r in enumerate(remainx) ]).flatten()
        lx =lx.flatten() - remainx/2
        ux =ux.flatten() + d



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


        print(urep)



        transition=np.zeros((un, xn, xn))
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

                P += (np.array([[reduce(operator.mul, p, 1) for p in itertools.product(*Pi)]]),)

                # print(sum( [reduce(operator.mul, p, 1) for p in itertools.product(*Pi)])) #check sums to one
                # sum(P)
            prob= np.concatenate(P, axis = 0)
            transition[u_index] = prob
            #print(np.concatenate(P, axis = 0))
            #print(np.sum(transition[u_index],axis=1))


        # compute u=0 probability distributions






        # for each dimension compute grid points
        # s = cell(n,1);  # enumerate
        # z_rep = cell(n,1); # grid values
        #
        #
        # for i= 1:n
        # nz(i)=ceil((boxmax_min(1,i)-boxmax_min(2,i))/(2*d));
        # z{i} = linspace(boxmax_min(2,i),boxmax_min(1,i),nz(i)+1); % boundaries of the partition sets for first state
        # z_rep{i} = z{1}(1:end-1) + diff(z{i})/2; % representative points for x1
        # end
        # % Add boundaries on the input ???
        # U_L=min(U.V);
        # U_H=max(U.V);
        # u = linspace(U_L,U_H,nu+1); % boundaries of the partition sets for input
        # u_diam = (U_H-U_L)/nu; % diameter for input
        # u_rep = u(1:nu) + diff(u)/2; % representative points for input
        # nu=length(u_rep);





        print('WARNING: UNVERIFIED implementation')
        transitions =transition
            #Transition probability matrices.

        mdp_grid = Markov(transitions, srep, urep,sedge)

        if verbose == True and n == 2:
            plt.scatter(grid[0].flatten(), grid[1].flatten(), label='finite states', color='k', s=10, marker="o")

            plt.xlabel('x')
            plt.ylabel('y')
            plt.legend()
            plt.show()

        return mdp_grid



