from Models.Linear import LTI
import numpy as np
from numpy import linalg as LA

import polytope as pc
import matplotlib.pyplot as plt
import itertools


A1 = np.array([[0,-0.8572],[1,1.857]])
print(LA.eig(A1) )
B1 = np.eye(2)# np.array([[-0.5343],[0.5523]])
#B1[1][1]=1
print(B1)

Bw1 = np.array([[-5.916e-3,-0.0564, 8.62e-3],[6.138e-3, 0.05852,-6.739e-3]])
C1 = np.array([0,1])


Q = np.array([[-0.08954,-0.07712]])
sys=LTI(A1,B1,C1,None)
poly=pc.box2poly(np.kron(np.ones((sys.dim, 1)), np.array([[-12, 12]])))

target=pc.box2poly(np.kron(np.ones((sys.dim, 1)), np.array([[-1.5, 1.5]])))


sys.setU(pc.box2poly(np.kron(np.ones((2, 1)), np.array([[-3,3]]))))
sys.setX(poly)

sys.setBw(np.array([[.9,0],[0.0,.9]]))


d=np.array([[.5,.5]])
mdp_grid = sys.abstract(d,un=5,verbose= False)

mdp_grid.settarget(Target=target)



#print(mdp_grid.V)
for i in range(0, 7):
    mdp_grid.reach_bell()

# figure, ax = plt.figure()
# figure.add_subplot('111')
# axes = figure.axes[0]
# axes.axis('equal')  # sets aspect ration to 1
# compute limits X
print(mdp_grid.srep)
print(mdp_grid.urep)
# state=np.empty(np.shape(mdp_grid.V))
# for s, sstate in enumerate(itertools.product(*mdp_grid.srep)):
#     state[s]=sstate[0]
#
#
xi, yi = np.meshgrid(*mdp_grid.srep)
print(mdp_grid.V.reshape(xi.shape, order='F'))


plt.pcolor(mdp_grid.sedge[0],mdp_grid.sedge[1], mdp_grid.V.reshape(xi.shape, order='F'))
plt.colorbar()


plt.xlim(np.array([-12,12]))
plt.ylim(np.array([-12,12]))
plt.show()

