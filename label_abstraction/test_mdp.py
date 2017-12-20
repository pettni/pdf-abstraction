from label_abstraction.mdp import *
import numpy as np

import __future__
def test_connection():

	T0 = np.array([[0, 1], [1, 0]])
	T1 = np.array([[1, 0], [0, 1]])
	mdp1 = MDP([T0, T1], input_fcn=lambda m: m, output_fcn=lambda n: set([n]), input_name='u', output_name='z')


	T0 = np.array([[1, 0], [1, 0]])
	T1 = np.array([[0, 1], [0, 1]])

	mdp2 = MDP([T0, T1], input_fcn=lambda m: m, output_fcn=lambda n: set([n]), input_name='z', output_name='y')

	pmdp = ProductMDP(mdp1, mdp2)

	np.testing.assert_almost_equal(pmdp.T(0).todense(), 
								   np.array([[0,0,0,1], [0,0,0,1], [1,0,0,0], [1,0,0,0]]))

	np.testing.assert_almost_equal(pmdp.T(1).todense(), 
								   np.array([[1,0,0,0], [1,0,0,0], [0,0,0,1], [0,0,0,1]]))

	vals1, _ = pmdp.solve_reach(accept=lambda s: s[0]==0 and s[1]==1)
	np.testing.assert_almost_equal(vals1, [0, 1, 0, 0])

	vals2, _ = pmdp.solve_reach(accept=lambda s: s[0]==0 and s[1]==0)
	np.testing.assert_almost_equal(vals2, [1, 1, 1, 1])


def test_reach():
	T0 = np.array([[0.5, 0.25, 0.25], [0, 1, 0], [0, 0, 1]])
	mdp = MDP([T0])
	
	V, _ = mdp.solve_reach(accept=lambda y: y==2)

	np.testing.assert_almost_equal(V, [0.5, 0, 1], decimal=4)

def test_mdp_dfsa():

	def output(n1):
		if n1 == 2:
			return set([1])
		else:
			return set([0])

	T0 = np.array([[0.5, 0.25, 0.25], [0, 1, 0], [0, 0, 1]])
	mdp = MDP([T0], output_fcn=output)

	T1 = np.array([[1, 0], [0, 1]])
	T2 = np.array([[0, 1], [0, 1]])
	fsa = MDP([T1, T2])

	prod = ProductMDP(mdp, fsa)

	V, _ = prod.solve_reach(accept=lambda y: y[1] == 1 )
	np.testing.assert_almost_equal(V,
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
	mdp = MDP([T0], output_fcn=output)

	T1 = np.array([[1, 0], [0, 1]])
	T2 = np.array([[0, 1], [0, 1]])
	fsa = MDP([T1, T2])

	prod = ProductMDP(mdp, fsa)

	V, _ = prod.solve_reach(accept=lambda y: y[1] == 1 )
	np.testing.assert_almost_equal(V,
								   [0.5, 1, 0, 1, 1, 1],
								   decimal=4)

def test_ltl_synth():

    T1 = np.array([[0.25, 0.25, 0.25, 0.25], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T2 = np.array([[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0.9, 0, 0, 0.1]])

    def output(n):
        # map Y1 -> 2^(2^AP)
        if n == 1:
            return set( ( ('s1',), ) )   # { {s1} }
        elif n == 3:
            return set( ( ('s2',), ) )   # { {s2} }
        else:
            return set( ( (), ), )		 # { { } }

    system = MDP([T1, T2], output_fcn = output, output_name ='ap')

    formula = '( ( F s1 ) & ( F s2 ) )'
    dfsa, init, final, _ = formula_to_mdp(formula)

    prod = ProductMDP(system, dfsa)

    V, _ = prod.solve_reach(accept=lambda s: s[1] in final)

    print('Value function',V )
    print(init, final)

    np.testing.assert_almost_equal(V[::4], [0.5, 0, 0, 0.5],
                                   decimal=4)
    # ISSSUE with this test,


def test_ltl_synth2():

	T1 = np.array([[0.25, 0.25, 0.25, 0.25], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
	T2 = np.array([[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0.9, 0, 0, 0.1]])

	def output(n):
		# map Y1 -> 2^(2^AP)
		if n == 1:
			return set( ( ('s1',), ) )   # { {s1} }
		elif n == 3:
			return set( ( ('s2',), ) )   # { {s2} }
		else:
			return set( ( (), ), )		 # { { } }

	system = MDP([T1, T2], output_fcn = output, output_name ='ap')

	formula = '( ( F s1 ) & ( F s2 ) )'

	pol = solve_ltl_cosafe(system, formula)


def test_ltl_until():

	formula = '( ( ! avoid U target ) & ( F avoid ) )'
	dfsa, _, _, _ = formula_to_mdp(formula)

	np.testing.assert_equal(len(dfsa), 4)