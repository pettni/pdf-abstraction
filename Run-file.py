"""
LTI Gaussian processes abstraction (basic)
Given
    -  x^+= A x + Bu + Ew
    -  z = Cx+Fv
Do :
    - Quantify eps,
    - Abstract to finite MDP
    Remark that since there is no order reduction,
     there is also no delta error.

Author Sofie Haesaert
"""

print('Import packages')
# Import packages:
from Models.Linear import LTI, POMDP
from ApprxSimulation.LTI_simrel import eps_err, tune_d, tune_dratio
import numpy as np
import polytope as pc
import matplotlib.pyplot as plt
from numpy import linalg as LA
from label_abstraction.mdp import *

print('Initialise values')
# Define the linear time invariant system
A = np.array([[0,-0.8572],[0.1,0.5]])
B = np.eye(2) #array([[1],[0.1]])
W = np.eye(2) # noise on transitions

# Accuracy
C = np.array([[1,0]]) # defines metric for error (||y_finite-y||< epsilon with y= cx   )
sys = LTI(A,B,C,None) # LTI system with   D = None

# define noise (Diagonal ONLY!)
sys.setBw(np.array([[.9,0],[0.0,.9]]))


# Sensor noise => pomdp
Cz = np.ones((1,2)) # what can be measured
V = np.eye(1)
    #  x^+= A x + Bu + w
    #  y = C x
    #  z = Cz x+v

P= np.eye(2)
mean = np.zeros((2,1))

pomdp = POMDP(sys, Cz, V, P, mean)
Pp = P
for i in range(15):
    x, Pu = pomdp.update(mean, Pp)
    x, Pp = pomdp.predict(np.array([[1],[1]]),x=mean, P=Pu)
    # updates of x,P

L, Pst = pomdp.kalman()

belief_mdp = pomdp.beliefmodel()


for i in range(15):
    (x,P) = belief_mdp.simulate(np.array([[1],[1]]))

test_mdp_dfsa()
#
# #
# # Define spaces
# poly = pc.box2poly(np.kron(np.ones((sys.dim, 1)), np.array([[-15, 15]])))
# sys.setX(poly) # X space
#
# sys.setU(pc.box2poly(np.kron(np.ones((sys.m, 1)), np.array([[-3,3]]))))
# # continuous set of inputs
# Dist = pc.box2poly(np.kron(np.ones((sys.dim, 1)), np.array([[-.1,.1]])))
#
# ## step 1: tune gridding ratio (find optimal horizontal d_1, and vertical d_2)
# # currently only available for 2D
# print('1.  Tune gridding ratio')
# d_opt, d_vals,eps_values = tune_dratio(sys)
# # d_opt has optimal ratio with grid diameter of 1
# # choose grid sizes (factor of d_opt)
# d = 1.2 * d_opt#  with distance measure
# print('Choose grid ribs as', d)
#
# ## step 2. grid and compute eps error
# # (tilde x-x)+= (A+BK) (tilde x-x) +r
# print('Grid Gaussian process')
# mdp_grid = sys.abstract(d, un=5,verbose = True)  # do the gridding
# print('--- done gridding')
#
#
#
# ## step 3. Do value iteration
# # (tilde x-x)+= (A+BK) (tilde x-x) +r
# print('Define target set')
# # Define target set in state space
# target=pc.box2poly(np.kron(np.ones((sys.dim, 1)), np.array([[-1.5, 1.5]])))
# mdp_grid.settarget(Target=target) # Add target set
#
# print('Compute rachability')
# for i in range(0, 7):
#     mdp_grid.reach_bell() # Bellman recursions
#
# xi, yi = np.meshgrid(*mdp_grid.srep)
#
# plt.pcolor(mdp_grid.sedge[0],mdp_grid.sedge[1], mdp_grid.V.reshape(xi.shape, order='F'))
# plt.colorbar()
# plt.xlim(np.array([-12,12]))
# plt.ylim(np.array([-12,12]))
# plt.show()
#
#
#
