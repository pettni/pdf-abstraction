import numpy as np

from mdp import *

def test_connection():

	T0 = np.array([[0, 1], [1, 0]])
	T1 = np.array([[1, 0], [0, 1]])
	mdp1 = MDP([T0, T1], input=lambda m: m, output=lambda n: n, input_name='u', output_name='z')


	T0 = np.array([[1, 0], [1, 0]])
	T1 = np.array([[0, 1], [0, 1]])

	mdp2 = MDP([T0, T1], input=lambda m: m, output=lambda n: n, input_name='z', output_name='y')

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
	
	vals = mdp.solve_reach(accept=lambda n: n==2)

	np.testing.assert_almost_equal(vals, [0.5, 0, 1], 1e-3)

def test_mdp_dfsa():

	def output(n1):
		# set-valued outputs
		if n1 == 2:
			return set([1])
		else:
			return set([0])

	T0 = np.array([[0.5, 0.25, 0.25], [0, 1, 0], [0, 0, 1]])
	mdp = MDP([T0], output=output)

	T1 = np.array([[1, 0], [0, 1]])
	T2 = np.array([[0, 1], [0, 1]])
	fsa = MDP([T1, T2])

	prod = MDP_DFSA(mdp, fsa)

	np.testing.assert_almost_equal(prod.solve_reach(accept=lambda n: n[1]==1),
								   [0.5, 1, 0, 1, 1, 1],
								   1e-3)

def test_mdp_mdp():

	T0 = np.array([[0.5, 0.25, 0.25], [0, 1, 0], [0, 0, 1]])
	mdp2 = MDP([T0])

	T1 = np.array([[1, 0], [0, 1]])
	T2 = np.array([[0, 1], [0, 1]])
	fsa2 = MDP([T1, T2])

	def out_conn2(n1):
		if n1 == 2:
			return 1
		else:
			return 0

	prod2 = ProductMDP(mdp2, fsa2, out_conn = out_conn2)

	np.testing.assert_almost_equal(prod2.solve_reach(accept=lambda n: n[1]==1),
								   [0.5, 1, 0, 1, 1, 1],
								   1e-3)
