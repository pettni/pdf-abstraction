import numpy as np
import scipy.sparse as sp
from itertools import product

from label_abstraction.ltl2mdp import *

class MDP(object):
  """Markov Decision Process"""
  def __init__(self, T, input_fcn=lambda m: m, output_fcn=lambda n: set([n]), 
               input_name='a', output_name='x'):
    '''
    Create an MDP

    Input arguments:
      T:  List of M stochastic transition matrices of size N x N such that T[m][n,n'] = P(n' | n, m).
          M is the number of actions and N the number of states.
      input:  input labeling function: U -> range(M) 
      output: output labeling function: range(N) -> 2^Y
      input_name: identifier for input state
      output_name: identifier for output state

    Alphabets:
      states: range(N)
      inputs: range(M)
      output alphabet: Y
      input  alphabet: U
    '''
    self.M = len(T)         # number of actions
    self.N = T[0].shape[1]  # number of states

    # Inputs are action labels
    self.input_fcn = input_fcn
    self.input_name = input_name


    # Outputs are state labels
    self.output_fcn = output_fcn
    self.output_name = output_name

    self.init = None
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

  def __len__(self):
    return self.N

  def T(self, m):
    '''transition matrix for action m'''
    return self.Tmat[m]

  def t(self, a, n, np):
    '''transition probability for action m and n -> np'''
    return self.T(a)[n, np]

  def input(self, u):
    '''index of input u'''
    return self.input_fcn(u)

  def output(self, n):
    '''output for state n''' 
    return self.output_fcn(n)

  def evolve(self, state, input):
    return self.T[input].dot(state)

  def solve_reach(self, accept, prec=1e-5):
    '''solve reachability problem
    Inputs:
    - accept: function range(N) -> {True, False} defining target set

    Outputs::
    - V: vector of length N representing probability to reach target for each state
    - pol: vector of length N representing optimal action m \in range(M)'''

    # todo: solve backwards reachability to constrain search

    is_accept = np.array([accept( list(self.output(n))[0] ) for n in range(self.N)])

    V = np.array(is_accept, dtype='d')
    pol = np.zeros(V.shape)

    while True:
      V_new_m = [self.T(m).dot(V) for m in range(self.M)]
      V_new = np.zeros(V.shape)
      for m in range(self.M):
        pol[np.nonzero(V_new < V_new_m[m])] = m
        V_new = np.maximum(V_new, V_new_m[m])
  
      V_new = np.maximum(V_new, is_accept)
      if np.max(np.abs(V_new - V)) < prec:
        break
      V = V_new

    return V, pol


class ProductMDP(MDP):
  """Non-deterministic Product Markov Decision Process"""
  def __init__(self, mdp1, mdp2):
    '''
    Connect mdp1 and mdp2 in series. It must hold that mdp1.Y \subset mdp2.U.
    The resulting mdp has inputs mdp1.U and output alphabet mdp2.Y
    '''

    self.deterministic = True

    # map range(N1) -> 2^range(N2)
    self.connection = lambda mdp1_n: set(map(mdp2.input, mdp1.output(mdp1_n)))

    # Check that connection is valid
    for n1 in range(mdp1.N):
      inputs_n1 = self.connection(n1)
      if not inputs_n1 <= set(range(mdp2.N)):
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

    self.output_name = '(%s, %s)' % (mdp1.output_name, mdp2.output_name) 

  # State ordering for mdp11 = (p1, ..., pN1) mdp12 = (q1, ..., qN2):
  #  (p1 q1) (p1 q2) ... (p1 qN2) (p2 q1) ... (pN1 qN2)
  def local_states(self, n):
    '''return indices in product'''
    return n // self.mdp2.N, n % self.mdp2.N

  def output(self, n):
    n1, n2 = self.local_states(n)
    return set(product(self.mdp1.output(n1), self.mdp2.output(n2)))

  def input(self, u):
    return self.mdp1.input(u)

  def T(self, m):
    if not self.deterministic:
      raise Exception('can not compute T matrix of nondeterministic product')
    ret =  sp.bmat([[self.mdp1.t(m, n, n_p)*self.mdp2.T( list(self.connection(n_p))[0] ) 
                     for n_p in range(self.mdp1.N)]
                    for n in range(self.mdp1.N)])
    ret.eliminate_zeros()
    return ret

  def t(self, m, n, n_p):
    if self.deterministic:
      raise Exception('can not compute transition probabilities in nondeterministic product')

    n1, n2 = self.local_states(n)
    n1p, n2p = self.local_states(n_p)

    return self.mdp1.t(m, n1, n1p) * self.mdp2.t(self.conn(n1p), n2, n2p)

  def solve_reach(self, accept, prec=1e-5):
    '''solve reachability problem
    Inputs:
    - accept: function range(N) -> {True, False} defining target set

    Outputs::
    - V: vector of length N representing probability to reach target for each state
    - pol: vector of length N representing optimal action m \in range(M)'''

    # todo: extend to longer connections via recursive computation of W
    # V(s1',s2',s3') -> V(s1',s2',s3) -> V(s1', s2, s3) -> V(s1, s2, s3)
    # where e.g.  V(s1',s2',s3) = \sum_{s3' \in y2(s2')} t3(m3, s3, s3') V(s1', s2', s3')

    is_accept = np.array([[accept((n, mu)) for n in range(self.mdp1.N)] for mu in range(self.mdp2.N)], dtype='d')
    # mu first dim, n second
    V = np.array(is_accept)

    Pol = np.zeros(is_accept.shape)

    while True:
      # Min over nondeterminism: W(mu,s') = min_{q \in y(s')} \sum_\mu' t(q,\mu,\mu') V(\mu', s')
      Wq_list = [self.mdp2.T(q).dot(V) for q in range(self.mdp2.M)]
      W = np.array([[min(Wq_list[q][mu, n] for q in self.connection(n)) for n in range(self.mdp1.N)]
                         for mu in range(self.mdp2.N)])
      # Max over actions: V_new(mu, s) = max_{m} \sum_s' t(m, s, s') W(mu, s')
      V_new_m = [self.mdp1.T(m).dot(W.transpose()).transpose() for m in range(self.M)]
      V_new = np.zeros(V.shape)
      for m in range(self.M):
        Pol[np.nonzero(V_new < V_new_m[m])] = m
        V_new = np.maximum(V_new, V_new_m[m])

      # Max over accepting state
      V_new = np.maximum(V_new, is_accept)

      if np.amax(np.abs(V_new - V)) < prec:
        break
      V = V_new

    return V.ravel(order='F'), Pol.ravel(order='F')


def formula_to_mdp(formula):
  fsa = Fsa()
  fsa.from_formula(formula)
  fsa.add_trap_state()

  # mapping state -> state index
  N = len(fsa.g)
  dict_fromstate = dict([(sstate, s) for s, sstate in enumerate(sorted(fsa.g.nodes()))])

  inputs = set.union(*[attr['input'] for _,_,attr in fsa.g.edges(data=True)])
  M = len(inputs)
  assert(inputs == set(range(M)))

  T = [np.zeros((N, N)) for m in range(M)]

  for (s1, s2, attr) in fsa.g.edges(data=True):
    for u in attr['input']:
      T[u][dict_fromstate[s1], dict_fromstate[s2]] = 1


  mdp = MDP(T, input_name='ap', input_fcn=fsa.bitmap_of_props,
         output_name='mu')

  init_states = set(map(lambda state: dict_fromstate[state], [state for (state, key) in fsa.init.items() if key == 1]))
  final_states = set(map(lambda state: dict_fromstate[state], fsa.final))

  mdp.init = list(init_states)

  # create map number => sets of atom propositions
  print(fsa.props.keys())
  # make tuple with on off
  props= tuple()
  for ap in fsa.props.keys():
      props+=([True, False],)
  dict_input2prop = dict()
  for status in product(*props):
      ap_set=tuple()
      for i,name in enumerate(fsa.props.keys()):
          if status[i]:
            ap_set += (name,)
      dict_input2prop[fsa.bitmap_of_props(ap_set)] = ap_set
  print(dict_input2prop)
  print(final_states)

  return mdp, init_states, final_states, dict_input2prop


class LTL_Policy(object):
  """control policy"""
  def __init__(self, dfsa, dfsa_init, pol, V):
    self.dfsa = dfsa
    self.dfsa_state = dfsa_init
    self.pol = pol
    self.V = V

  def report_system_output(self, out):
    dfsa_action = self.dfsa.input_fcn(out)
    self.dfsa_state = self.dfsa.evolve(self.dfsa_state, dfsa_action)

  def get_system_input(self, syst_state):
    prod_state = syst_state * self.dfsa.M + self.dfsa_state
    return self.pol(prod_state)


def solve_ltl_cosafe(mdp, formula):
  '''synthesize a policy that maximizes the probability of
     satisfaction of formula
     Inputs:
      - mdp: a MDP or ProductMDP with output alphabet Y
      - formula: a syntactically cosafe LTL formula over Y

     Example: If Y = {'s1', 's2'}, the MDP output map should take 
     values in 2^2^{'s1', 's2'} 

     Outputs:
      - pol: a Policy maximizing the probability of enforcing formula''' 

  dfsa, init, final,aux = formula_to_mdp(formula)
  prod = ProductMDP(mdp, dfsa)
  V, pol = prod.solve_reach(accept=lambda s: s[1] in final)

  return LTL_Policy(dfsa, list(init)[0], pol, V)

