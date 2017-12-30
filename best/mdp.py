import numpy as np
import scipy.sparse as sp
import operator
from sparse_tensor import sparse_tensor

import time

from best import prod

def idx_to_midx(idx, n_list):
  # index to multiindex
  assert idx >= 0
  assert idx < prod(n_list)

  return tuple(idx % prod(n_list[i:]) / prod(n_list[i + 1:]) for i in range(len(n_list)))


def midx_to_idx(midx, n_list):
  # multiindex to index
  assert len(midx) == len(n_list)
  assert all(midx[i] < n_list[i] for i in range(len(midx)))
  assert all(midx[i] >= 0 for i in range(len(midx)))

  return sum(midx[i] * prod(n_list[i+1:]) for i in range(len(n_list)))

class MDP(object):
  """Markov Decision Process"""
  def __init__(self, T, input_fcn=lambda m: m, output_fcn=lambda n: n, 
               input_name='a', output_name='x'):
    '''
    Create an MDP

    Input arguments:
      T:  List of M stochastic transition matrices of size N x N such that T[m][n,n'] = P(n' | n, m).
          M is the number of actions and N the number of states.
      input:  input labeling function: U -> range(M) 
      output: output labeling function: range(N) -> Y
      input_name: identifier for input state
      output_name: identifier for output state

    Alphabets:
      states: range(N)
      inputs: range(M)
      output alphabet: Y
      input  alphabet: U
    '''

    # Inputs are action labels
    self.input_fcn = input_fcn
    self.input_name = input_name

    # Outputs are state labels
    self.output_fcn = output_fcn
    self.output_name = output_name

    # Transition matrices for each axis
    self.Tmat_csr = [None,] * len(T)
    self.Tmat_coo = [None,] * len(T)

    for m in range(len(T)):
      self.Tmat_csr[m] = sp.csr_matrix(T[m])  # convert to sparse format
      self.Tmat_coo[m] = sp.coo_matrix(T[m])

    self.check()

  @property
  def N(self):
    return self.Tmat_coo[0].shape[1]

  @property
  def M(self):
    return len(self.Tmat_coo)

  def check(self):
    for m in range(self.M):
      t = self.T(m)
      if not t.shape == (self.N, self.N):
        raise Exception('matrix not square')

      if not np.all(np.abs(t.dot(np.ones([self.N,1])) - np.ones([self.N,1])) < 1e-7 ):
        raise Exception('matrix not stochastic')

  def __str__(self):
    ret = 'MDP: {0} inputs "{2}" --> {1} outputs "{3}"' \
          .format(self.M, self.N, self.input_name, self.output_name)
    return ret

  def __len__(self):
    return self.N

  def nnz(self):
    '''total number of stored transitions'''
    return sum(self.Tcoo(m).nnz for m in range(self.M))

  def sparsity(self):
    '''percentage of transitions'''
    return float(self.nnz()) / (self.N**2 * self.M)

  def global_state(self, n):
    return n

  def local_states(self, n):
    return n

  def resolve_connection(self, new_mdp, connection):
    new_conn_list = [[] for k in range(self.N)]
    new_det = True

    for n in range (self.N):
      inputs = connection(self.output(n))
      for inp in inputs:
        new_conn_list[n].append( new_mdp.input(inp) )

      if not set(new_conn_list[n]) <= set(range(new_mdp.M)):
        raise Exception('invalid connection')
      if len(new_conn_list[n]) == 0:
        raise Exception('empty connection')
      if len(new_conn_list[n]) > 1:
        new_det = False
    return new_conn_list, new_det

  def product(self, new_mdp, connection):
    if isinstance(new_mdp, ProductMDP2):
      Error('not implemented')

    new_conn_list = np.zeros([new_mdp.M, self.N])
    new_det = True
    
    for n in range (self.N):
      u_list = connection(self.output(n))

      if len(u_list) == 0:
        raise Exception('empty connection')

      if len(u_list) > 1:
        new_det = False

      for u in u_list:
        inp = new_mdp.input(u)
        if inp < 0 or inp >= new_mdp.M:
          raise Exception('invalid connection')
        new_conn_list[inp, n] = 1

    return ProductMDP2([self, new_mdp], [new_conn_list], [new_det])

  def T(self, m):
    '''transition matrix for action m'''
    return self.Tmat_csr[m]

  def Tcoo(self, m):
    '''transition matrix for action in coo format'''
    return self.Tmat_coo[m]

  def t(self, a, n, np):
    '''transition probability for action m and n -> np'''
    return self.T(a)[n, np]

  def input(self, u):
    '''index of input u'''
    return self.input_fcn(u)

  def output(self, n):
    '''output for state n''' 
    return self.output_fcn(n)

  def evolve(self, state, m):
    return self.T(m).transpose().dot(state)

  def solve_reach(self, accept, prec=1e-5):
    '''solve reachability problem
    Inputs:
    - accept: function range(N) -> {True, False} defining target set

    Outputs::
    - V: vector of length N representing probability to reach target for each state
    - pol: vector of length N representing optimal action m \in range(M)'''

    # todo: solve backwards reachability to constrain search

    is_accept = np.array([accept( self.output(n) ) for n in range(self.N)])

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

class ParallelMDP(MDP):

  def __init__(self, mdplist):
    '''
    Connect mdp1 and mdp2 in parallel
    The resulting mdp has inputs mdp1.U x mdp2.U and output alphabet mdp1.Y x mdp2.Y
    '''

    self.mdplist = mdplist

    self.computedT_coo = [None for m in range(self.M)]
    self.computedT_csr = [None for m in range(self.M)]

  @property
  def N(self):
    return reduce(operator.mul, [self.mdplist[i].N for i in range(len(self.mdplist))], 1)

  @property
  def M(self):
    return reduce(operator.mul, [self.mdplist[i].M for i in range(len(self.mdplist))], 1)

  @property
  def input_name(self):
    return '(' + ', '.join(self.mdplist[i].input_name for i in range(len(self.mdplist))) + ')'

  @property
  def output_name(self):
    return '(' + ', '.join(self.mdplist[i].output_name for i in range(len(self.mdplist))) + ')'

  def global_state(self, n_loc):
    '''local state n_loc to global n'''
    n_list = [mdp.N for mdp in self.mdplist]
    n = sum(self.mdplist[i].global_state(n_loc[i]) * prod(n_list[i+1:]) 
            for i in range(len(self.mdplist)))
    return n

  def local_states(self, n):
    '''global n to local state n_loc'''
    n_list = [mdp.N for mdp in self.mdplist]
    return tuple( self.mdplist[i].local_states(n % prod(n_list[i:]) / prod(n_list[i + 1:]))
            for i in range(len(n_list)))

  def local_controls(self, m):
    '''global to local controls'''
    m_list = [mdp.M for mdp in self.mdplist]
    return tuple(m % prod(m_list[i:]) / prod(m_list[i + 1:])
            for i in range(len(m_list)))

  def output(self, n):
    '''output of global n'''
    n_list = [mdp.N for mdp in self.mdplist]
    loc_idx = tuple(n % prod(n_list[i:]) / prod(n_list[i + 1:])
                    for i in range(len(n_list)))
    return tuple(mdpi.output(ni) for (mdpi, ni) in zip(self.mdplist, loc_idx))

  def input(self, u):
    m_list = [mdp.M for mdp in self.mdplist]
    m = sum(self.mdplist[i].input(u[i]) * prod(m_list[i+1:]) for i in range(len(self.mdplist)))
    return m

  def t(self, m, n, n_p):
    loc_n = self.local_states(n)
    loc_n_p = self.local_states(n_p)
    loc_m = self.local_controls(m)
    return prod(mdp.t(mi, ni, ni_p) for mdp, mi, ni, ni_p in zip(self.mdplist, loc_m, loc_n, loc_n_p))

  def computeTm(self, m):
    loc_m = self.local_controls(m)
    Tm_coo = reduce(sp.kron, (mdp.Tcoo(mi) for (mdp, mi) in zip(self.mdplist, loc_m)), 1)
    Tm_coo.eliminate_zeros()

    self.computedT_coo[m] = Tm_coo
    self.computedT_csr[m] = Tm_coo.tocsr()

  def T(self, m):
    if self.computedT_csr[m] is None:
      self.computeTm(m)
    
    return self.computedT_csr[m]

  def Tcoo(self, m):
    if self.computedT_coo[m] is None:
      self.computeTm(m)

    return self.computedT_coo[m]


class ProductMDP(MDP):
  """Non-deterministic Product Markov Decision Process"""
  def __init__(self, mdp1, mdp2, connection=None):
    '''
    Connect mdp1 and mdp2 in series. It must hold that mdp1.Y \subset mdp2.U.
    The resulting mdp has inputs mdp1.U and output alphabet mdp2.Y
    '''

    self.deterministic = True

    # map range(N1) -> 2^range(M2)
    if connection:
      n1m2_conn = lambda mdp1_n: set(map(mdp2.input, connection(mdp1.output(mdp1_n) ) ))
    else:
      n1m2_conn = lambda mdp1_n: set([mdp2.input(mdp1.output(mdp1_n))])

    # compute connections
    self.conn_list = [[m2 for m2 in n1m2_conn(n1)] for n1 in range(mdp1.N)]

    # Check that connection is valid
    for n1 in range(mdp1.N):
      if not self.conn_list[n1] <= set(range(mdp2.M)):
        raise Exception('invalid connection')
      if len(self.conn_list[n1]) > 1:
        self.deterministic = False

    if not self.deterministic:
      print('warning: creating nondeterministic product')

    # mdp1.check()
    # mdp2.check()

    self.mdp1 = mdp1
    self.mdp2 = mdp2

    self.M = mdp1.M
    self.N = mdp1.N * mdp2.N

    self.input_name = mdp1.input_name
    self.output_name = '(%s, %s)' % (mdp1.output_name, mdp2.output_name) 

    self.computedT_csr = [None for m in range(self.M)]
    self.computedT_coo = [None for m in range(self.M)]

  # State ordering for mdp11 = (p1, ..., pN1) mdp12 = (q1, ..., qN2):
  #  (p1 q1) (p1 q2) ... (p1 qN2) (p2 q1) ... (pN1 qN2)
  def local_states(self, n):
    '''return indices in product'''
    return (self.mdp1.local_states(n // self.mdp2.N), self.mdp2.local_states(n % self.mdp2.N))

  def global_state(self, n):
    return self.mdp2.global_state(n[1]) + self.mdp2.N * self.mdp1.global_state(n[0])

  def output(self, n):
    return (self.mdp1.output(n // self.mdp2.N), self.mdp2.output(n % self.mdp2.N))

  def input(self, u):
    return self.mdp1.input(u)

  def T(self, m):
    if not self.deterministic:
      raise Exception('can not compute T matrix of nondeterministic product')

    if self.computedT[m] is None:
      Tm =  sp.bmat([[self.mdp1.t(m, n, n_p)*self.mdp2.T( self.conn_list[n_p][0] ) 
                       for n_p in range(self.mdp1.N)]
                      for n in range(self.mdp1.N)])
      Tm.eliminate_zeros()
      self.computedT[m] = Tm

    return self.computedT[m]

  def computeTm(self, m):
    if not self.deterministic:
      raise Exception('can not compute T matrix of nondeterministic product')

    # desired matrix has blocks of form [ T1[m, s1, s1p] * T2[c(c1p), :, :] ]

    print "computing Tm"
    start = time.time()

    T1coo = self.mdp1.Tcoo(m)

    mat_list = [[sp.coo_matrix((self.mdp2.N, self.mdp2.N)) for np in range(self.mdp1.N)] 
                for n in range(self.mdp1.N)]
    # loop over non-zero entries in T1(m)
    for (ni, npi, pri) in zip(T1coo.row, T1coo.col, T1coo.data):
      mat_list[ni][npi] = pri * self.mdp2.Tcoo( self.conn_list[npi][0] )

    Tm_coo = sp.bmat(mat_list)
    Tm_coo.eliminate_zeros()

    self.computedT_coo[m] = Tm_coo
    self.computedT_csr[m] = Tm_coo.tocsr()

    print "computed Tm in ", time.time()-start

  def T(self, m):
    if self.computedT_csr[m] is None:
      self.computeTm(m)
    
    return self.computedT_csr[m]

  def Tcoo(self, m):
    if self.computedT_coo[m] is None:
      self.computeTm(m)

    return self.computedT_coo[m]

  def t(self, m, n, n_p):
    if not self.deterministic:
      raise Exception('can not compute transition probabilities in nondeterministic product')

    # n1, n2 = self.local_states(n)
    # n1p, n2p = self.local_states(n_p)
    return self.T(m)[n,n_p]
    # return self.mdp1.t(m, n1, n1p) * self.mdp2.t( list(self.connection(n1p))[0], n2, n2p)

  def solve_reach(self, accept, maxiter=np.Inf, prec=1e-5):
    '''solve reachability problem
    Inputs:
    - accept: function range(N) -> {True, False} defining target set

    Outputs::
    - V: vector of length N representing probability to reach target for each state
    - pol: vector of length N representing optimal action m \in range(M)'''

    # todo: extend to longer connections via recursive computation of W
    # V(s1',s2',s3') -> V(s1',s2',s3) -> V(s1', s2, s3) -> V(s1, s2, s3)
    # where e.g.  V(s1',s2',s3) = \sum_{s3' \in y2(s2')} t3(m3, s3, s3') V(s1', s2', s3')

    # todo: use sparse V and compute only in neighborhood of positivity

    is_accept = np.array([[accept((n, mu)) for n in range(self.mdp1.N)] for mu in range(self.mdp2.N)], dtype='d')
    # mu first dim, n second
    V = np.array(is_accept)

    Pol = np.zeros(is_accept.shape, dtype=int)

    it = 0
    start = time.time()

    while it < maxiter:

      print('iteration {}, time {}'.format(it, time.time()-start))
      # Min over nondeterminism: W(mu,s') = min_{q \in y(s')} \sum_\mu' t(q,\mu,\mu') V(\mu', s')
      Wq_list = [self.mdp2.T(q).dot(V) for q in range(self.mdp2.M)]
      W = np.array([[min(Wq_list[q][mu, n] for q in self.conn_list[n]) for n in range(self.mdp1.N)]
                         for mu in range(self.mdp2.N)])
      # Max over actions: V_new(mu, s) = max_{m} \sum_s' t(m, s, s') W(mu, s')
      V_new_m = [self.mdp1.T(m).dot(W.transpose()).transpose() for m in range(self.M)]
      V_new = np.zeros(V.shape)
      for m in range(self.M):
        Pol[np.nonzero(V < V_new_m[m])] = m
        V_new = np.maximum(V_new, V_new_m[m])

      # Max over accepting state
      V_new = np.maximum(V_new, is_accept)

      if np.amax(np.abs(V_new - V)) < prec:
        break
      V = V_new

      it += 1

    print('finished after {} iterations and {}s'.format(it, time.time()-start))
    return V.ravel(order='F'), Pol.ravel(order='F')


class ProductMDP2(MDP):
  """Non-deterministic Product Markov Decision Process"""
  def __init__(self, mdplist, conn_list, det_list):

    self.mdplist = mdplist
    self.conn_list = conn_list
    self.det_list = det_list

  @property
  def N(self):
    return prod(self.mdplist[i].N for i in range(len(self.mdplist)))

  @property
  def M(self):
    return self.mdplist[0].M

  @property
  def input_name(self):
    return self.mdplist[0].input_name

  @property
  def output_name(self):
    return '(' + ', '.join(self.mdplist[i].output_name 
                           for i in range(len(self.mdplist))) + ')'

  @property
  def N_list(self):
    return [self.mdplist[i].N for i in range(len(self.mdplist))]


  def global_state(self, n_loc):
    '''local state n_loc to global n'''
    n_list = [mdp.N for mdp in self.mdplist]   
    n = sum(self.mdplist[i].global_state(n_loc[i]) * prod(n_list[i+1:]) 
            for i in range(len(self.mdplist)))
    return n

  def local_states(self, n):
    '''global n to local state n_loc'''
    n_list = [mdp.N for mdp in self.mdplist]
    return tuple( self.mdplist[i].local_states(n % prod(n_list[i:]) / prod(n_list[i + 1:]))
            for i in range(len(n_list)))

  def output(self, n):
    '''output of global n'''
    n_list = [mdp.N for mdp in self.mdplist]
    loc_idx = tuple(n % prod(n_list[i:]) / prod(n_list[i + 1:])
                    for i in range(len(n_list)))
    return tuple(mdpi.output(ni) for (mdpi, ni) in zip(self.mdplist, loc_idx))

  def input(self, u):
    return mdplist[0].input(u)

  def product(self, new_mdp, connection):
    ''' Attach a product mdp '''
    if isinstance(new_mdp, ProductMDP2):
      Error('not implemented')

    new_conn_list = np.zeros([new_mdp.M] + self.N_list)
    new_det = True

    for n in range (self.N):
      midx = idx_to_midx(n, self.N_list)
      u_list = connection(self.output(n))

      if len(u_list) == 0:
        raise Exception('empty connection')

      if len(u_list) > 1:
        new_det = False

      for u in u_list:
        inp = new_mdp.input(u)
        if inp < 0 or inp >= new_mdp.M:
          raise Exception('invalid connection')
        new_conn_list[inp][midx] = 1

        # conn_list_a.append(inp)
        # new_conn_list[n].append( new_mdp.input(inp) )

    return ProductMDP2(self.mdplist + [new_mdp],
                       self.conn_list + [new_conn_list],
                       self.det_list + [new_det])


  def solve_reach(self, accept, maxiter=np.Inf, prec=1e-5):
    '''solve reachability problem
    Inputs:
    - accept: function defining target set

    Outputs::
    - V: vector of length N representing probability to reach target for each state
    - pol: vector of length N representing optimal action m \in range(M)'''

    # todo: use sparse V and compute only in neighborhood of positivity

    V = np.zeros(self.N_list)

    V_accept = np.zeros(self.N_list)
    for n in range(self.N):
      midx = idx_to_midx(n, self.N_list)
      V_accept[midx] = accept(midx) 

    it = 0
    start = time.time()

    while it < maxiter:
      print('iteration {}, time {}'.format(it, time.time()-start))
      
      W = np.fmax(V, V_accept)

      for i in range(len(self.mdplist)-1, 0, -1):
        # carry out computation for i = n-1 ... 1

        # for each action to system i
        Wq_list = np.array([sparse_tensor(self.mdplist[i].T(q), W, i) 
                            for q in range(self.mdplist[i].M)])

        # add dummy ones before minimizing
        newdim = Wq_list.shape[:i+1] + tuple (1 for k in range(i, len(self.mdplist)))
        reps = tuple(1 for k in range(i+1)) + Wq_list.shape[i+1:]

        Wq_dummy = (1 - self.conn_list[i-1]).reshape(newdim)  # promote to same dimension as Wq_list
        Wq_dummy = np.tile(Wq_dummy, reps)                    # tile to make same size as Wq_list
        Wq_list += Wq_dummy

        # # Min over nondeterminism
        W = Wq_list.min(axis=0)

      # Max over actions: V_new(mu, s) = max_{m} \sum_s' t(m, s, s') W(mu, s')
      V_new_m = np.array([sparse_tensor(self.mdplist[0].T(m), W, 0) 
                          for m in range(self.M)])
      V_new = V_new_m.max(axis=0)

      if np.amax(np.abs(V_new - V)) < prec:
        break
      V = V_new

      it += 1

    # Policy
    Pol = V_new_m.argmax(axis=0)

    return V.ravel(), Pol.ravel()