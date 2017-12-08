import numpy as np
import scipy.sparse as sp
from itertools import product


class MDP(object):
	"""Markov Decision Process"""
	def __init__(self, T, input=lambda m: m, output=lambda n: set([n]), 
							 input_name='a', output_name='x'):
		'''
		Create an MDP

		Input arguments:
			T:	List of M stochastic transition matrices of size N x N such that T[m][n,n'] = P(n' | n, m).
					M is the number of actions and N the number of states.
			input: 	input labeling function: range(M) -> U
			output: output labeling function: range(N) -> 2^Y
			input_name:	identifier for input state
			output_name: identifier for output state

		Alphabets:
			states: range(N)
			inputs: range(M)
			output alphabet: Y
			input labels: U
		'''
		self.M = len(T)			# number of actions
		self.N = T[0].shape[1]  	# number of states

		# Inputs are action labels
		self.inputs = {}
		for m in range(self.M):
			self.inputs[m] = input(m)
		self.input_name = input_name

		# Outputs are state labels
		self.outputs = {}
		for n in range(self.N):
			self.outputs[n] = output(n)
		self.output_name = output_name

		# Transition matrices for each axis
		self.Tmat = [None,] * self.M

		for m in range(self.M):
			self.Tmat[m] = sp.csr_matrix(T[m])  # convert to sparse format

		self.check()

	def check(self):
		for m in range(self.M):
			t = self.T(m)
			if not t.shape == (self.N, self.N):
				raise Exception('matrix not square')

			if not np.all(np.abs(t.dot(np.ones([self.N,1])) - np.ones([self.N,1])) < 1e-7 ):
				raise Exception('matrix not stochastic')

		for n in range(self.N):
			if not type(self.output(n) is set):
				raise Exception('MDP outputs must be of type set')

	def __str__(self):
		ret = 'MDP: %d states "%s" and %d inputs "%s"' % (self.N, self.output_name, self.M, self.input_name)
		return ret

	def T(self, m):
		'''transition matrix for action m'''
		return self.Tmat[m]

	def t(self, a, n, np):
		'''transition probability for action m and n -> np'''
		return self.T(a)[n, np]

	def output(self, n):
		'''output for state n''' 
		return self.outputs[n]

	def solve_reach(self, accept):
		'''compute reachability for target defined by accept : outputs -> true/false function'''

		# todo: solve backwards reachability to constrain search

		is_accept = np.array([accept( list(self.output(n))[0] ) for n in range(self.N)])

		values = np.array(is_accept, dtype='d')

		while True:
			new_values = np.amax([self.T(a).dot(values) for a in range(self.M)], axis=0)
			new_values = np.fmax(new_values, is_accept)
			if np.max(np.abs(new_values - values)) < 1e-5:
				break
			values = new_values

		return values


class ProductMDP(MDP):
	"""Non-deterministic Product Markov Decision Process"""
	def __init__(self, mdp1, mdp2, out_conn = lambda y: y):
		'''
		Connect mdp1 and mdp2 in series via out_conn: mdp1.Y -> mdp2.U.
		The resulting mdp has inputs mdp1.U and output alphabet mdp2.Y
		'''

		self.deterministic = True

		# this maps mdp1.X -> 2^mdp2.U
		self.connection = lambda mdp1_n: set(map(out_conn, mdp1.output(mdp1_n)))

		# Check that connection is valid
		for n1 in range(mdp1.N):
			inputs_n1 = self.connection(n1)
			if not inputs_n1 <= set(mdp2.inputs.values()):
				raise Exception('invalid connection')
			if len(inputs_n1) > 1:
				self.deterministic = False

		if not self.deterministic:
			print('warning: creating nondeterministic product')

		mdp1.check()
		mdp2.check()

		self.mdp1 = mdp1
		self.mdp2 = mdp2

		self.M = mdp1.M
		self.N = mdp1.N * mdp2.N

		# Map states of mdp1 to actions of mdp2

		self.input_name = mdp1.input_name
		self.inputs = mdp1.inputs

		self.output_name = '(%s, %s)' % (mdp1.output_name, mdp2.output_name) 

	# State ordering for mdp11 = (p1, ..., pN1) mdp12 = (q1, ..., qN2):
	#  (p1 q1) (p1 q2) ... (p1 qN2) (p2 q1) ... (pN1 qN2)
	def local_states(self, n):
		'''return indices in product'''
		return n // self.mdp2.N, n % self.mdp2.N

	def output(self, n):
		n1, n2 = self.local_states(n)
		return set(product(self.mdp1.output(n1), self.mdp2.output(n2)))

	def T(self, m):
		if not self.deterministic:
			raise Exception('can not compute T matrix of nondeterministic product')
		print (self.connection(0))
		ret =  sp.bmat([[self.mdp1.t(m, n, n_p)*self.mdp2.T( list(self.connection(n_p))[0] ) 
									   for n_p in range(self.mdp1.N)]
			      			  for n in range(self.mdp1.N)])
		ret.eliminate_zeros()
		return ret

	def t(self, m, n, n_p):
		if self.deterministic:
			raise Exception('can not compute transition probabilities in product')

		n1, n2 = self.local_states(n)
		n1p, n2p = self.local_states(n_p)

		return self.mdp1.t(m, n1, n1p) * self.mdp2.t(self.conn(n1p), n2, n2p)

	def solve_reach(self, accept):
		'''solve reachability problem'''

		is_accept = sp.csr_matrix([[accept((n, mu)) for n in range(self.mdp1.N)] for mu in range(self.mdp2.N)], dtype='d')
		# mu first dim, n second
		V = sp.csr_matrix(is_accept)

		while True:

			# Min over nondeterminism: W(mu,s') = min_{q \in y(s')} \sum_\mu' t(q,\mu,\mu') V(\mu', s')
			Wq_list = [self.mdp2.T(q).dot(V) for q in range(self.mdp2.M)]
			W = sp.csr_matrix([[min(Wq_list[q][mu, n] for q in self.connection(n)) for n in range(self.mdp1.N)]
												 for mu in range(self.mdp2.N)])

			# Max over actions: V_new = max_{m} \sum_s' t(m, s, s') W(mu, s')
			V_new = W.dot(self.mdp1.T(0).transpose())
			for m in range(1, self.M):
				V_new = V_new.maximum(W.dot(self.mdp1.T(m)).transpose() )
			
			# Max over accepting state
			V_new = V_new.maximum(is_accept)

			if np.amax(np.abs(V_new - V)) < 1e-6:
				break
			V = V_new

		return V.todense().getA().ravel(order='F')
