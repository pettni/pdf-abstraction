import numpy as np
import polytope as pc

from best.models.lti import LTI
from best.abstraction.gridding import Abstraction, LTIAbstraction

def test_grid():
  abstr = Abstraction([-1, -1], [1, 1], [5, 5])
  for s in range(5*5):
    np.testing.assert_equal(s, abstr.x_to_s(abstr.s_to_x(s)))

  xx = 2 * np.random.rand(2,10) - 1
  for i in range(10):
    x = xx[:,i].flatten()
    s = abstr.x_to_s(x)
    xc = abstr.s_to_x(s)

    np.testing.assert_array_less( np.abs(xc - x),  0.2)


def test_tranformation():
  dim = 2
  A = np.eye(2) #np.array([[.9,-0.32],[0.1,0.9]])
  B = np.eye(dim)  #array([[1], [0.1]])
  W = np.array([[0,0],[0,0.4]]) #2*Tr.dot(np.eye(dim)).dot(Tr)  # noise on transitions
   
  # Accuracy
  C = np.array([[1, 0],[0,1]])  # defines metric for error (||y_finite-y||< epsilon with y= cx   )

  sys_lti = LTI(A, B, C, None, W=W)  # LTI system with   D = None

  X = pc.box2poly(np.kron(np.ones((sys_lti.dim, 1)), np.array([[-10, 10]])))
  U = pc.box2poly(np.kron(np.ones((sys_lti.m, 1)), np.array([[-1, 1]])))
  sys_lti.setU(U) # continuous set of inputs
  sys_lti.setX(X) # X space

  d = np.array([[1.],[1.]])

  abstr = LTIAbstraction(sys_lti, d, un=4)

  xx = 20 * np.random.rand(2,10) - 10
  for i in range(10):
    x = xx[:,i].reshape((2,1))
    s_ab = abstr.closest_abstract(x)

    x_out = abstr.mdp.output(s_ab)

    np.testing.assert_equal(x_out[0], s_ab )
    np.testing.assert_array_less( np.abs(x_out[1] - x), d/2 * (1 + 1e-5) )


def test_related():
  dim = 2
  A = np.eye(2) #np.array([[.9,-0.32],[0.1,0.9]])
  B = np.eye(dim)  #array([[1], [0.1]])
  W = np.array([[1,0],[0,1]]) #2*Tr.dot(np.eye(dim)).dot(Tr)  # noise on transitions
   
  # Accuracy
  C = np.array([[1, 0],[0,1]])  # defines metric for error (||y_finite-y||< epsilon with y= cx   )

  sys_lti = LTI(A, B, C, None, W=W)  # LTI system with   D = None

  X = pc.box2poly(np.kron(np.ones((sys_lti.dim, 1)), np.array([[-10, 10]])))
  U = pc.box2poly(np.kron(np.ones((sys_lti.m, 1)), np.array([[-1, 1]])))
  sys_lti.setU(U) # continuous set of inputs
  sys_lti.setX(X) # X space

  d = np.array([[1.],[1.]])

  abstr = LTIAbstraction(sys_lti, d, un=3)

  for s in np.arange(0, len(abstr), 7):
    np.testing.assert_equal( abstr.closest_abstract(abstr.s_to_x(s) ), s )

