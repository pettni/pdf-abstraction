import numpy as np

from mdp import *

def test_connection():

	T0 = np.array([[0, 1], [1, 0]])
	T1 = np.array([[1, 0], [0, 1]])
	mdp1 = MDP([T0, T1], input=lambda m: m, output=lambda n: set([n]), input_name='u', output_name='z')


	T0 = np.array([[1, 0], [1, 0]])
	T1 = np.array([[0, 1], [0, 1]])

	mdp2 = MDP([T0, T1], input=lambda m: m, output=lambda n: set([n]), input_name='z', output_name='y')

	pmdp = ProductMDP(mdp1, mdp2, out_conn=lambda z: z)

	np.testing.assert_almost_equal(pmdp.T(0).todense(), 
								   np.array([[0,0,0,1], [0,0,0,1], [1,0,0,0], [1,0,0,0]]))

	np.testing.assert_almost_equal(pmdp.T(1).todense(), 
								   np.array([[1,0,0,0], [1,0,0,0], [0,0,0,1], [0,0,0,1]]))

	vals1 = pmdp.solve_reach(accept=lambda s: s[0]==0 and s[1]==1)
	np.testing.assert_almost_equal(vals1, [0, 1, 0, 0])

	vals2 = pmdp.solve_reach(accept=lambda s: s[0]==0 and s[1]==0)
	np.testing.assert_almost_equal(vals2, [1, 1, 1, 1])


def test_reach():
	T0 = np.array([[0.5, 0.25, 0.25], [0, 1, 0], [0, 0, 1]])
	mdp = MDP([T0])
	
	vals = mdp.solve_reach(accept=lambda y: y==2)

	np.testing.assert_almost_equal(vals, [0.5, 0, 1], decimal=4)

def test_mdp_dfsa():

	def output(n1):
		if n1 == 2:
			return set([1])
		else:
			return set([0])

	T0 = np.array([[0.5, 0.25, 0.25], [0, 1, 0], [0, 0, 1]])
	mdp = MDP([T0], output=output)

	T1 = np.array([[1, 0], [0, 1]])
	T2 = np.array([[0, 1], [0, 1]])
	fsa = MDP([T1, T2])

	prod = ProductMDP(mdp, fsa)

	np.testing.assert_almost_equal(prod.solve_reach(accept=lambda y: y[1] == 1 ),
								   [0.5, 1, 0, 1, 1, 1],
								   decimal=4)

def test_mdp_dfsa_nondet():

	def output(n1):
		if n1 == 2:
			return set([1])
		elif n1 == 1:
			return set([1,0])
		else:
			return set([0])

	T0 = np.array([[0.5, 0.25, 0.25], [0, 1, 0], [0, 0, 1]])
	mdp = MDP([T0], output=output)

	T1 = np.array([[1, 0], [0, 1]])
	T2 = np.array([[0, 1], [0, 1]])
	fsa = MDP([T1, T2])

	prod = ProductMDP(mdp, fsa)

	np.testing.assert_almost_equal(prod.solve_reach(accept=lambda y: y[1] == 1 ),
								   [0.5, 1, 0, 1, 1, 1],
								   decimal=4)
