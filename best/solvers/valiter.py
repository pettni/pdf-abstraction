'''Module for table value iteration'''
import time
import copy
import numpy as np

from best import DTYPE, DTYPE_ACTION, DTYPE_OUTPUT
from best.logic.translate import formula_to_pomdp

def solve_reach(network, accept, horizon=np.Inf, delta=0, prec=1e-5, verbose=False):
  '''solve reachability problem
  Inputs:
  - network: a POMDPNetwork
  - accept: function defining target set
  - horizon: reachability horizon (standard is infinite-horizon)
  - delta: failure probability in each step
  - prec: termination tolerance (inifinite-horizon case)

  Outputs::
  - val_list: array [V0 V1 .. VT] of value functions

  The infinite-horizon problem has a stationary value function and policy. In this
  case the return argument has length 2, i.e. val_list = [V0 VT]
  '''

  V_accept = accept.astype(DTYPE, copy=False)

  it = 0
  start = time.time()

  V = np.fmax(V_accept, np.zeros(network.N, dtype=DTYPE))

  val_list = []
  pol_list = []
  val_list.insert(0, V)

  while it < horizon:

    if verbose:
      print('iteration {}, time {}'.format(it, time.time()-start))

    # Calculate Q(u_free, x)
    Q = network.bellman(V).reshape((-1,) + network.N)

    # Max over free actions
    P_new = np.unravel_index(Q.argmax(axis=0).astype(DTYPE_ACTION, copy=False), network.M)
    V_new = Q.max(axis=0)

    # Max over accept set
    V_new = np.fmax(V_accept, np.maximum(V_new - delta, 0))

    if horizon < np.Inf:
      val_list.insert(0, V_new)
      pol_list.insert(0, P_new)

    if horizon == np.Inf and np.amax(np.abs(V_new - V)) < prec:
      val_list.insert(0, V_new)
      pol_list.insert(0, P_new)
      break

    V = V_new
    it += 1

  print('finished after {}s and {} iterations'.format(time.time()-start, it))

  return val_list, pol_list


def solve_ltl_cosafe(network, formula, predicates, delta=0., horizon=np.Inf, verbose=False):
  '''synthesize a policy that maximizes the probability of
     satisfaction of formula
     Inputs:
      - network: a POMDPNetwork 
      - formula: a syntactically cosafe LTL formula over AP
      - predicates: list of triples ('output', 'ap', output -> 2^{0,1}) defining atomic propositions
                        if an atomic proposition depends on several outputs, add an intermediate logic gate

     Example: If AP = {'s1', 's2'}, then connection(x) should be a 
     value in 2^2^{'s1', 's2'} 

     Outputs:
      - pol: a LTL_Policy maximizing the probability of enforcing formula''' 

  dfsa, dfsa_init, dfsa_final = formula_to_pomdp(formula)

  network_copy = copy.deepcopy(network)

  network_copy.add_pomdp(dfsa)
  for output, ap, conn in predicates:
    network_copy.add_connection(output, ap, conn)

  Vacc = np.zeros(network_copy.N)
  Vacc[...,list(dfsa_final)[0]] = 1

  val, pol = solve_reach(network_copy, Vacc, delta=delta, horizon=horizon, verbose=verbose)
  
  return LTL_Policy(dfsa.input_names, dfsa._Tmat_csr, list(dfsa_init)[0], dfsa_final, val, pol)


class LTL_Policy(object):
  """control policy"""
  def __init__(self, proplist, dfsa_Tlist, dfsa_init, dfsa_final, val, pol):
    '''create a control policy object'''
    self.proplist = proplist
    self.dfsa_Tlist = dfsa_Tlist
    self.dfsa_init = dfsa_init
    self.dfsa_final = dfsa_final
    self.val = val
    self.pol = pol

    self.dfsa_state = self.dfsa_init

  def reset(self):
    '''reset controller'''
    self.dfsa_state = self.dfsa_init

  def report_aps(self, aps):
    '''report atomic propositions to update internal controller state'''
    dfsa_action = tuple(int(ap in aps) for ap in self.proplist)
    row = self.dfsa_Tlist[dfsa_action].getrow(self.dfsa_state)
    self.dfsa_state = row.indices[0]

  def __call__(self, syst_state, t=0):
    '''get input from policy'''
    if t >= len(self.val)-1:
      print('Warning: t={} larger than horizon {}. Setting t={}'.format(t, len(self.val)-1, len(self.val)-2))
      t = len(self.val)-2
    joint_state = tuple(syst_state) + (self.dfsa_state,)

    u = tuple(self.pol[t][m][joint_state] for m in range(len(self.pol[t])))

    return u, self.val[t][joint_state]

  def finished(self):
    '''check if policy reached target'''
    return self.dfsa_state in self.dfsa_final
