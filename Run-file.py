from Models.Linear import LTI
import numpy as np
from numpy import linalg as LA

import polytope as pc
import matplotlib.pyplot as plt
import itertools


A1 = np.array([[0,-0.8572],[1,1.857]])
B1 = np.array([[-0.5343],[0.5523]])

Bw1 = np.array([[-5.916e-3,-0.0564, 8.62e-3],[6.138e-3, 0.05852,-6.739e-3]])
C1 = np.array([0,1])


Q = np.array([[-0.08954,-0.07712]])
sys=LTI(A1,B1,C1,None)
poly=pc.box2poly(np.kron(np.ones((sys.dim, 1)), np.array([[-22, 22]])))

sys.setU(pc.box2poly(np.kron(np.ones((sys.m, 1)), np.array([[-3,3]]))))
sys.setX(poly)

sys.setBw(np.array([[.9,0],[0.0,.9]]))


d=np.array([[.5,.5]])
mdp_grid = sys.abstract(d,un=5,verbose= False)

target=pc.box2poly(np.kron(np.ones((sys.dim, 1)), np.array([[-1.5, 1.5]])))
mdp_grid.settarget(Target=target)

for i in range(0, 7):
    mdp_grid.reach_bell()


xi, yi = np.meshgrid(*mdp_grid.srep)
print(mdp_grid.V.reshape(xi.shape, order='F'))


plt.pcolor(mdp_grid.sedge[0],mdp_grid.sedge[1], mdp_grid.V.reshape(xi.shape, order='F'))
plt.colorbar()


plt.xlim(np.array([-12,12]))
plt.ylim(np.array([-12,12]))
plt.show()

