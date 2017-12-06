"""
LTI Gaussian processes abstraction (basic)
Given
    -  x^+= A x + Bu + w
    -  w ~ N(0, aI)
Do :
    - Quantify eps, del
    - Abstract to finite MDP
Author Sofie
"""


# Import packages:
from Models.Linear import LTI
import numpy as np
import polytope as pc
import matplotlib.pyplot as plt

# Define the linear time invariant system
A = np.array([[0,-0.8572],[1,1.857]])
B = np.array([[-0.5343],[0.5523]])



sys = LTI(A,B,None,None) # LTI system with C=none and D = None
# define noise (Diagonal ONLY!)
sys.setBw(np.array([[.9,0],[0.0,.9]]))

# Define spaces
poly = pc.box2poly(np.kron(np.ones((sys.dim, 1)), np.array([[-22, 22]])))
sys.setX(poly) # X space

sys.setU(pc.box2poly(np.kron(np.ones((sys.m, 1)), np.array([[-3,3]]))))
# continuous set of inputs


# *Grid space
d = np.array([[.5,.5]]) #  with distance measure
mdp_grid = sys.abstract(d,un=5,verbose= False)  #  do the gridding


# Define target set in state space
target=pc.box2poly(np.kron(np.ones((sys.dim, 1)), np.array([[-1.5, 1.5]])))

mdp_grid.settarget(Target=target) # Add target set

for i in range(0, 7):
    mdp_grid.reach_bell() # Bellman recursions

xi, yi = np.meshgrid(*mdp_grid.srep)

plt.pcolor(mdp_grid.sedge[0],mdp_grid.sedge[1], mdp_grid.V.reshape(xi.shape, order='F'))
plt.colorbar()
plt.xlim(np.array([-12,12]))
plt.ylim(np.array([-12,12]))
plt.show()

