import numpy as np
import scipy.sparse as sp
import operator

import time

from best import *

DTYPE = np.float32
DTYPE_ACTION = np.uint8

class MDP(object):
  """Markov Decision Process"""
  def __init__(self, T, input_fcn=None, output_fcn=None, 
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
      self.Tmat_csr[m] = sp.csr_matrix(T[m], dtype=DTYPE)
      self.Tmat_coo[m] = sp.coo_matrix(T[m], dtype=DTYPE)

    self.check()

  @property
  def N(self):
    return self.Tmat_coo[0].shape[1]

  @property
  def N_list(self):
    return (self.N,)

  @property
  def M(self):
    return len(self.Tmat_coo)

  def check(self):
    for m in range(self.M):
      t = self.T(m)
      if not t.shape == (self.N, self.N):
        raise Exception('matrix not square')

      if not np.all(np.abs(t.dot(np.ones([self.N,1])) - np.ones([self.N,1])) < 1e-5 ):
        raise Exception('matrix not stochastic')

  def __str__(self):
    ret = 'MDP: {0} inputs "{2}" --> {1} outputs "{3}"' \
          .format(self.M, self.N, self.input_name, self.output_name)
    return ret

  def __len__(self):
    return self.N

  def prune(self, tresh = 1e-8):
    # remove transitions with probability less than tresh and renormalize
    for m in range(self.M):
      data = self.Tmat_csr[m].data
      indices = self.Tmat_csr[m].indices
      indptr = self.Tmat_csr[m].indptr
      data[np.nonzero(data < tresh)] = 0

      new_mat = sp.csr_matrix((data, indices, indptr), shape=self.Tmat_csr[m].shape)

      # diagonal matrix with row sums
      norms = new_mat.dot( np.ones(new_mat.shape[1]) )
      norms_mat = sp.coo_matrix((1/norms, (range(new_mat.shape[1]), range(new_mat.shape[1])))) 

      self.Tmat_csr[m] = norms_mat.dot(new_mat)
      self.Tmat_coo[m] = sp.coo_matrix(self.Tmat_csr[m])

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
    if isinstance(new_mdp, ProductMDP):
      raise Exception('not implemented')

    new_conn_list = np.zeros([new_mdp.M, self.N], dtype=bool)
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

    return ProductMDP([self, new_mdp], [new_conn_list], [new_det])

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
    if self.input_fcn is None:
      return u
    return self.input_fcn(u)

  def output(self, n):
    '''output for state n''' 
    if self.output_fcn is None:
      return n
    return self.output_fcn(n)

  def evolve(self, state, m):
    return self.T(m).transpose().dot(state)

  def bellman(self, W):
    '''compute V_{ux} = \sum_{x'} T_{uxx'} W_x' '''
    return np.stack([sparse_tensordot(self.T(m), W, 0) for m in range(self.M)])

  def solve_reach_constrained(self, Vacc0, Vcon0, treshhold, horizon, verbose=False):
    '''
      Compute max  P(reach Vacc0)
              s.t. P(reach Vcon0) >= treshold
    '''
    Vacc = np.zeros(self.N_list, dtype=DTYPE)
    Vcon = np.zeros(self.N_list, dtype=DTYPE)

    val_list = []
    pol_list = []

    it = 0
    start = time.time()
    while it < horizon:
      if verbose:
        print('iteration {}, time {}'.format(it, time.time()-start))

      # Iterate constraint
      Vcon_new_m = self.bellman(np.fmax(Vcon, Vcon0.astype(DTYPE)))
      Vcon = Vcon_new_m.max(axis = 0)

      # Max action over those that satisfy constraint
      Vacc_new_m = self.bellman(np.fmax(Vacc, Vacc0.astype(DTYPE)))

      Vcon_gr = Vcon_new_m >= treshhold  # to bool

      Vacc = (Vacc_new_m * Vcon_gr).max(axis = 0)

      # Subtract to satisfy constraint even if Vacc=0
      pol = (Vacc_new_m * Vcon_gr - 2*(1-Vcon_gr)).argmax(axis = 0).astype(DTYPE_ACTION, copy=False)

      val_list.insert(0, Vacc)
      pol_list.insert(0, pol)

      it += 1

    print('finished after {}s and {} iterations'.format(time.time()-start, it))

    return val_list, pol_list

  def solve_reach(self, accept, horizon=np.Inf, delta=0, prec=1e-5, verbose=False):
    '''solve reachability problem
    Inputs:
    - accept: function defining target set
    - horizon: reachability horizon (standard is infinite-horizon)
    - delta: failure probability in each step
    - prec: termination tolerance (inifinite-horizon case)

    Outputs::
    - val_list: array [V0 V1 .. VT] of value functions
    - pol_list: array [P0 P1 .. PT] of value functions

    The infinite-horizon problem has a stationary value function and policy. In this
    case the return arguments have length 1, i.e. val_list = [V0], pol_list = [P0].
    '''

    if type(accept).__module__ == np.__name__:
      # accept already given
      V_accept = accept.astype(DTYPE, copy=False)
    else:
      # we got a function
      V_accept = np.zeros(self.N_list, dtype=DTYPE)
      for n in range(self.N):
        midx = idx_to_midx(n, self.N_list)
        V_accept[midx] = accept(self.output(n)) 

    it = 0
    start = time.time()

    V = np.zeros(self.N_list, dtype=DTYPE)

    val_list = []
    pol_list = []

    while it < horizon:
      if verbose:
        print('iteration {}, time {}'.format(it, time.time()-start))
      
      V_new_m = self.bellman(np.fmax(V, V_accept))
      V_new = np.maximum(V_new_m.max(axis=0) - delta, 0)

      if horizon < np.Inf and it < horizon-1:
        val_list.insert(0, V_new)
        pol_new = V_new_m.argmax(axis=0).astype(DTYPE_ACTION, copy=False)
        pol_list.insert(0, pol_new)

      if np.amax(np.abs(V_new - V)) < prec:
        break
      V = V_new

      it += 1

    val_list.insert(0, V_new)
    pol_new = V_new_m.argmax(axis=0).astype(DTYPE_ACTION, copy=False)
    pol_list.insert(0, pol_new)

    print('finished after {}s and {} iterations'.format(time.time()-start, it))

    return val_list, pol_list


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

  @property
  def N_list(self):
    return [self.mdplist[i].N for i in range(len(self.mdplist))]

  def global_state(self, n_loc):
    '''local state n_loc to global n'''

    if isinstance(n_loc, int):
      # for recursive calls
      return n_loc

    if not len(self.N_list) == len(n_loc):
      raise Exception('invalid length')

    n = sum(self.mdplist[i].global_state(n_loc[i]) * prod(self.N_list[i+1:]) 
            for i in range(len(self.mdplist)))
    return n

  def local_states(self, n):
    '''global n to local state n_loc'''
    return tuple( self.mdplist[i].local_states(n % prod(self.N_list[i:]) / prod(self.N_list[i + 1:]))
            for i in range(len(self.N_list)))

  def local_controls(self, m):
    '''global to local controls'''
    m_list = [mdp.M for mdp in self.mdplist]
    return tuple(m % prod(m_list[i:]) / prod(m_list[i + 1:])
            for i in range(len(m_list)))

  def output(self, n):
    '''output of global n'''
    loc_idx = tuple(n % prod(self.N_list[i:]) / prod(self.N_list[i + 1:])
                    for i in range(len(self.N_list)))
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
    Tm_coo = reduce(sp.kron, (mdp.Tcoo(mi) for (mdp, mi) in zip(self.mdplist, loc_m)), np.array([1], dtype=DTYPE) )
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

  def T(self, m):
    if not all(self.det_list):
      raise Exception('can not compute transition matrix of nondeterministic product')

    Tret = self.mdplist[0].Tcoo(m)

    for i in range(len(self.mdplist)-1):
      N = Tret.shape[0]

      mdp1 = self.mdplist[i]
      mdp2 = self.mdplist[i+1]

      conn = self.conn_list[i]
      
      # loop over non-zero entries in T1(m)
      mat_list = [[sp.coo_matrix((mdp2.N, mdp2.N), dtype=DTYPE) for n2 in range(N)] 
                  for n1 in range(N)]

      for (ni, npi, pri) in zip(Tret.row, Tret.col, Tret.data):

        midx = idx_to_midx(npi, self.N_list[:i+1])
        ai = np.nonzero(conn[(slice(None, mdp2.M, None),) + midx].flatten())[0][0]
        mat_list[ni][npi] = pri * mdp2.Tcoo( ai )

      Tret = sp.bmat(mat_list)

    Tret.eliminate_zeros()
    return Tret

  def global_state(self, n_loc):
    '''local state n_loc to global n'''

    if isinstance(n_loc, int):
      # for recursive calls
      return n_loc

    if not len(self.N_list) == len(n_loc):
      raise Exception('invalid length')

    n = sum(self.mdplist[i].global_state(n_loc[i]) * prod(self.N_list[i+1:]) 
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
    if isinstance(new_mdp, ProductMDP):
      raise Exception('not implemented')

    new_conn_list = np.zeros([new_mdp.M] + self.N_list, dtype=bool)
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

    return ProductMDP(self.mdplist + [new_mdp],
                       self.conn_list + [new_conn_list],
                       self.det_list + [new_det])

  def bellman(self, W):
    ''' sequentially compute V_{ux} = \sum_{x'} T_{uxx'} W_{x'} for a serial MDP'''

    for i in range(len(self.mdplist)-1, 0, -1):

      if not self.det_list[i-1]:
        # connection is nondeterministic
    
        # for each action to system i
        Wq_list = np.stack([sparse_tensordot(self.mdplist[i].T(m), W, i) 
                          for m in range(self.mdplist[i].M)])

        # add dummy ones before minimizing
        newdim = Wq_list.shape[:i+1] + tuple (1 for k in range(i, len(self.mdplist)))
        reps = tuple(1 for k in range(i+1)) + Wq_list.shape[i+1:]

        Wq_dummy = (1 - self.conn_list[i-1]).reshape(newdim)  # promote to same dimension as Wq_list
        Wq_dummy = np.tile(Wq_dummy, reps)                    # tile to make same size as Wq_list
        Wq_list += Wq_dummy

        # Min over nondeterminism
        W = Wq_list.min(axis=0)

      else: 
        # connection is deterministic
        dummy = self.conn_list[i-1]
        while len(dummy.shape) < 1 + len(self.mdplist):
          # promote to higher dim to ensure broadcasting is done correctly
          dummy = dummy[...,np.newaxis]

        # element-wise product
        W = reduce(np.add, 
                   (dummy[m] * sparse_tensordot(self.mdplist[i].T(m), W, i) 
                    for m in range(self.mdplist[i].M)),
                   np.zeros(self.N_list, dtype=DTYPE))

    # Max over actions: V_new(mu, s) = max_{m} \sum_s' t(m, s, s') W(mu, s')
    V_new_m = np.stack([sparse_tensordot(self.mdplist[0].T(m), W, 0) 
                        for m in range(self.M)])

    return V_new_m
