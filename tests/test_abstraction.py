from best.abstraction import LTIAbstraction
from Models.Linear import LTI

import numpy as np
import polytope as pc

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
		print x
		s_ab = abstr.closest_abstract(x)

		x_out = abstr.mdp.output(s_ab)

		np.testing.assert_equal(x_out[0], s_ab )
		np.testing.assert_array_less( np.abs(x_out[1] - x), d/2 * (1 + 1e-5) )