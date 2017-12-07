import numpy as np
import scipy.sparse as sp

class MDP(object):
	"""Markov Decision Process"""
	def __init__(self, T, input=lambda m: m, output=lambda n: n, 
							 input_name='a', output_name='x'):
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

		# todo: do updates locally for pre of non-zero elements only

		# todo: add non-determinism..
		is_accept = np.array([accept(self.output(n)) for n in range(self.N)])
		values = np.array(is_accept, dtype='d')

		while True:
			new_values = np.amax([self.T(a).dot(values) for a in range(self.M)], axis=0)
			new_values = np.fmax(new_values, is_accept)
			if np.max(np.abs(new_values - values)) < 1e-5:
				break
			values = new_values

		return values


class ProductMDP(MDP):
	"""Product Markov Decision Process"""
	def __init__(self, mdp1, mdp2, out_conn = lambda m: m):

		# Check that connection is valid
		if not set([out_conn(mdp1.output(n1)) for n1 in range(mdp1.N)]) <= set(mdp2.inputs.values()):
			raise Exception('invalid connection')
		mdp1.check()
		mdp2.check()

		self.mdp1 = mdp1
		self.mdp2 = mdp2

		self.M = mdp1.M
		self.N = mdp1.N * mdp2.N

		self.connection = lambda n_p : out_conn(mdp1.output(n_p))

		self.input_name = mdp1.input_name
		self.inputs = mdp1.inputs

		self.output_name = '(%s, %s)' % (mdp1.output_name, mdp2.output_name) 

		self.check()

	# State ordering for MDP1 = (p1, ..., pN1) MDP2 = (q1, ..., qN2):
	#  (p1 q1) (p1 q2) ... (p1 qN2) (p2 q1) ... (pN1 qN2)
	def local_states(self, n):
		'''return indices in product'''
		return n // self.mdp2.N, n % self.mdp2.N

	def output(self, n):
		n1, n2 = self.local_states(n)
		return (self.mdp1.output(n1), self.mdp2.output(n2))

	def T(self, m):
		ret =  sp.bmat([[self.mdp1.t(m, n, n_p)*self.mdp2.T(self.connection(n_p)) 
									   for n_p in range(self.mdp1.N)]
			      			  for n in range(self.mdp1.N)])
		ret.eliminate_zeros()
		return ret

	def t(self, m, n, n_p):
		n1, n2 = self.local_states(n)
		n1p, n2p = self.local_states(n_p)

		return self.mdp1.t(m, n1, n1p) * self.mdp2.t(self.conn(n1p), n2, n2p)

class MDP_DFSA(object):
	"""Non-deterministic Product Markov Decision Process"""
	def __init__(self, mdp, fsa, conn = lambda m: m):

		# Check that connection is valid
		for n1 in range(mdp.N):
			if not set(conn(o) for o in mdp.output(n1)) <= set(fsa.inputs.values()):
				raise Exception('invalid connection')
		mdp.check()
		fsa.check()

		# todo: assert that fsa is really an fsa

		self.mdp = mdp
		self.fsa = fsa

		self.M = mdp.M
		self.N = mdp.N * fsa.N

		self.conn = conn

		self.input_name = mdp.input_name
		self.inputs = mdp.inputs

		self.output_name = '(%s, %s)' % (mdp.output_name, fsa.output_name) 

	# State ordering for MDP1 = (p1, ..., pN1) MDP2 = (q1, ..., qN2):
	#  (p1 q1) (p1 q2) ... (p1 qN2) (p2 q1) ... (pN1 qN2)
	def local_states(self, n):
		'''return indices in product'''
		return n // self.mdp.N, n % self.mdp.N

	def output(self, n):
		n1, n2 = self.local_states(n)
		return (self.mdp.output(n1), self.fsa.output(n2))

	def solve_reach(self, accept):
		'''solve reachability problem'''

		is_accept = np.array([accept((n, mu)) for n in range(self.mdp.N) for mu in range(self.fsa.N)])
		V = np.array(is_accept, dtype='d')

		def vslice(n_p):
			return V[n_p*self.fsa.N:(n_p+1)*self.fsa.N]

		# Too ugly...
		while True:
			new_V = np.zeros(self.N)
			for n in range(self.mdp.N):
				for mu in range(self.fsa.N):
					maxval = 0
					for m in range(self.M):
						valm = sum(min(self.mdp.t(m, n, n_p) * self.fsa.T(self.conn(out)).getrow(mu).dot(vslice(n_p))  
									         for out in self.mdp.output(n_p))
							         for n_p in range(self.mdp.N))
						maxval = max(maxval, valm)
					new_V[n * self.fsa.N + mu] = maxval
			
			new_V = np.fmax(is_accept, new_V)

			if np.max(np.abs(new_V - V)) < 1e-5:
				break
			V = new_V

		return V
