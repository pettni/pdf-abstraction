from best.mdp import MDP, ProductMDP, ParallelMDP
from best.ltl import solve_ltl_cosafe, formula_to_mdp
import numpy as np

import __future__
def test_connection():

	T0 = np.array([[0, 1], [1, 0]])
	T1 = np.array([[1, 0], [0, 1]])
	mdp1 = MDP([T0, T1], input_fcn=lambda m: m, output_fcn=lambda n: n, input_name='u', output_name='z')


	T0 = np.array([[1, 0], [1, 0]])
	T1 = np.array([[0, 1], [0, 1]])

	mdp2 = MDP([T0, T1], input_fcn=lambda m: m, output_fcn=lambda n: n, input_name='z', output_name='y')

	pmdp = mdp1.product(mdp2, lambda n1: set([n1]))

	np.testing.assert_almost_equal(pmdp.T(0).todense(), 
								   np.array([[0,0,0,1], [0,0,0,1], [1,0,0,0], [1,0,0,0]]))

	np.testing.assert_almost_equal(pmdp.T(1).todense(), 
								   np.array([[1,0,0,0], [1,0,0,0], [0,0,0,1], [0,0,0,1]]))

	vals1, _ = pmdp.solve_reach(accept=lambda s: s[0]==0 and s[1]==1)
	np.testing.assert_almost_equal(vals1[0], [0, 1, 0, 0])

	vals2, _ = pmdp.solve_reach(accept=lambda s: s[0]==0 and s[1]==0)
	np.testing.assert_almost_equal(vals2[0], [1, 1, 1, 1])

def test_connection():

	T0 = np.array([[0.5, 0.5], [0, 1.]])
	T1 = np.array([[0.2, 0.8], [1, 0]])
	T2 = np.array([[0.2, 0.8], [1, 0]])

	mdp1 = MDP([T0, T1, T2])
	mdp2 = MDP([T0, T1, T2])
	mdp3 = MDP([T0, T1, T2])

	prod = mdp1.product(mdp2, connection = lambda n: set([1]))
	prod = prod.product(mdp3, connection = lambda n: set([2]))

	pT = np.kron(np.kron(T0, T1), T2)

	np.testing.assert_almost_equal(prod.T(0).todense(), pT)

def test_prune():
	T0 = np.array([[0.5, 0.05, 0.45], [0, 1, 0], [0, 0.01, 0.99]])
	mdp = MDP([T0])
	mdp.prune(thresh=0.06)

	TprunedA = np.array([[0.5/0.95, 0, 0.45/0.95], [0, 1, 0], [0, 0, 1]])

	Tpruned = mdp.T(0).todense()
	print Tpruned
	np.testing.assert_almost_equal(Tpruned, TprunedA)

def test_reach():
	T0 = np.array([[0.5, 0.25, 0.25], [0, 1, 0], [0, 0, 1]])
	mdp = MDP([T0])
	
	V, _ = mdp.solve_reach(accept=lambda y: y==2)

	np.testing.assert_almost_equal(V[0], [0.5, 0, 1], decimal=4)

def test_mdp_dfsa():

	def output(n1):
		if n1 == 2:
			return 1
		else:
			return 0

	T0 = np.array([[0.5, 0.25, 0.25], [0, 1, 0], [0, 0, 1]])
	mdp = MDP([T0], output_fcn=output)

	T1 = np.array([[1, 0], [0, 1]])
	T2 = np.array([[0, 1], [0, 1]])
	fsa = MDP([T1, T2])

	connection = lambda n1: set([n1])

	prod = mdp.product(fsa, connection)

	V, _ = prod.solve_reach(accept=lambda y: y[1] == 1 )
	np.testing.assert_almost_equal(V[0],
								   [[0.5, 1], [0, 1], [1, 1]],
								   decimal=4)

def test_mdp_dfsa_nondet():

	def connection(n1):
		if n1 == 2:
			return set([1])
		elif n1 == 1:
			return set([1, 0])
		else:
			return set([0])

	T0 = np.array([[0.5, 0.25, 0.25], [0, 1, 0], [0, 0, 1]])
	mdp = MDP([T0])

	T1 = np.array([[1, 0], [0, 1]])
	T2 = np.array([[0, 1], [0, 1]])
	fsa = MDP([T1, T2])

	prod = mdp.product(fsa, connection)

	V, _ = prod.solve_reach(accept=lambda y: y[1] == 1 )
	np.testing.assert_almost_equal(V[0],
								   [[0.5, 1], [0, 1], [1, 1]],
								   decimal=4)

def test_ltl_synth():

    T1 = np.array([[0.25, 0.25, 0.25, 0.25], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    T2 = np.array([[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0.9, 0, 0, 0.1]])

    def connection(n):
        # map Y1 -> 2^(2^AP)
        if n == 1:
            return set( ( ('s1',), ) )   # { {s1} }
        elif n == 3:
            return set( ( ('s2',), ) )   # { {s2} }
        else:
            return set( ( (), ), )		 # { { } }

    system = MDP([T1, T2])

    formula = '( ( F s1 ) & ( F s2 ) )'
    dfsa, init, final, _ = formula_to_mdp(formula)

    prod = system.product(dfsa, connection)

    V, _ = prod.solve_reach(accept=lambda s: s[1] in final)

    np.testing.assert_almost_equal(V[0][:,0], [0.5, 0, 0, 0.5],
                                   decimal=4)


def test_ltl_synth2():

	T1 = np.array([[0.25, 0.25, 0.25, 0.25], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
	T2 = np.array([[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0.9, 0, 0, 0.1]])

	def connection(n):
		# map Y1 -> 2^(2^AP)
		if n == 1:
			return set( ( ('s1',), ) )   # { {s1} }
		elif n == 3:
			return set( ( ('s2',), ) )   # { {s2} }
		else:
			return set( ( (), ), )		 # { { } }

	system = MDP([T1, T2])

	formula = '( ( F s1 ) & ( F s2 ) )'

	pol = solve_ltl_cosafe(system, formula, connection)


def test_ltl_until():

	formula = '( ( ! avoid U target ) & ( F avoid ) )'
	dfsa, _, _, _ = formula_to_mdp(formula)

	np.testing.assert_equal(len(dfsa), 4)


def test_parallel():

	T0 = np.eye(3)
	T1 = np.array([[0,0.5,0.5], [0,1,0], [0,0,1]])

	def output_fcn(n):
	    if n == 0:
	        return 'init'    # label unknown
	    if n == 1:
	        return 'safe'    # can traverse region
	    if n == 2:
	        return 'unsafe'  # can not traverse region
	    
	map1 = MDP([T0, T1], input_fcn=lambda meas1: meas1, input_name='meas1',
	                     output_fcn=output_fcn, output_name='label1')

	map2 = MDP([T0, T1], input_fcn=lambda meas2: meas2, input_name='meas2',
	                     output_fcn=output_fcn, output_name='label2')

	map3 = MDP([T0, T1], input_fcn=lambda meas3: meas3, input_name='meas3',
	                     output_fcn=output_fcn, output_name='label3')

	prod = ParallelMDP([map1, map2, map3])

	for i1 in range(2):
		for i2 in range(2):
			for i3 in range(2):
				np.testing.assert_equal((i1,i2,i3), prod.local_controls(prod.input((i1,i2,i3))))


	for k in range(8):
		T = prod.T(k).todense()
		for i in range(27):
			for j in range(27):
				np.testing.assert_almost_equal(T[i,j], prod.t(k,i,j))

def test_reach_finitetime():

	T0 = np.array([[0.9, 0, 0.1], [0, 1, 0], [0, 0, 1]])
	T1 = np.array([[0, 0.5, 0.5], [0, 1, 0], [0, 0, 1]])

	mdp = MDP([T0, T1])

	accept = lambda n: n == 2

	vlist, plist = mdp.solve_reach(accept, horizon=3)

	np.testing.assert_almost_equal(vlist[0][0], 0.1 + 0.9*0.1 + 0.9**2*0.5)
	np.testing.assert_almost_equal(vlist[1][0], 0.1 + 0.9*0.5)
	np.testing.assert_almost_equal(vlist[2][0], 0.5)

	np.testing.assert_almost_equal(plist[0][0], 0)
	np.testing.assert_almost_equal(plist[1][0], 0)
	np.testing.assert_almost_equal(plist[2][0], 1)

def test_reach_constrained():
	T0 = np.array([[0, 0.5, 0.5], [0, 1, 0], [0, 0, 1]])

	mdp = MDP([T0])

	Vacc = np.array([0, 1, 1])
	Vcon = np.array([0, 1, 0])

	vlist, plist = mdp.solve_reach_constrained(Vacc, Vcon, 0.4, 4)

	np.testing.assert_almost_equal(vlist[0], [1, 1, 0])

	vlist, plist = mdp.solve_reach_constrained(Vacc, Vcon, 0.6, 4)

	np.testing.assert_almost_equal(vlist[0], [0, 1, 0])


def test_reach_constrained2():
	T0 = np.array([[0, 0.98, 0, 0, 0.01, 0.01], [0, 0, 0.98, 0, 0.01, 0.01], [0, 0, 0, 0.98, 0.01, 0.01], [0, 0, 0, 0.98, 0.01, 0.01], [0,0,0,0,1,0], [0,0,0,0,0,1]])
	T1 = np.array([[0, 0.5, 0, 0, 0.5, 0], [0, 0, 0.5, 0, 0.5, 0], [0, 0, 0, 0, 1, 0], [0, 0, 0, 0.99, 0.01, 0], [0,0,0,0,1, 0], [0,0,0,0,0,1]])

	mdp = MDP([T1, T0])

	Vacc = np.array([0, 0, 0, 1, 1, 0])
	Vcon = np.array([0, 0, 0, 1, 0, 0])

	vlist, plist = mdp.solve_reach_constrained(Vacc, Vcon, 0.94, 4)

	np.testing.assert_array_less([0.95, 0.95, 0.95, 0.999, -0.0001, -0.0001], vlist[0])

	vlist, plist = mdp.solve_reach_constrained(Vacc, Vcon, 0.95, 4)

	np.testing.assert_array_less(vlist[0], [0.01, 1.01, 1.01, 1.01, 0.01, 0.01])


