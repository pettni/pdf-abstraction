import numpy as np
import scipy.sparse as sp
from itertools import product

from best.fsa import Fsa
from best.mdp import MDP, ProductMDP

def formula_to_mdp(formula):
  '''convert a co-safe LTL formula to a DFSA represented as a   
  special case of an MPD'''
  
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

  return mdp, init_states, final_states, fsa.props


def solve_ltl_cosafe(mdp, formula, connection, horizon=np.Inf, delta=0., verbose=False):
  '''synthesize a policy that maximizes the probability of
     satisfaction of formula
     Inputs:
      - mdp: a MDP or ProductMDP with output alphabet Y
      - formula: a syntactically cosafe LTL formula over AP
      - connection: mapping Y -> 2^2^AP

     Example: If AP = {'s1', 's2'}, then connection(x) should be a 
     value in 2^2^{'s1', 's2'} 

     Outputs:
      - pol: a Policy maximizing the probability of enforcing formula''' 

  dfsa, dfsa_init, dfsa_final, proplist = formula_to_mdp(formula)

  prod_mdp = mdp.product(dfsa, connection)

  Vacc = np.zeros(prod_mdp.N_list)
  Vacc[...,list(dfsa_final)[0]] = 1

  val, pol = prod_mdp.solve_reach(Vacc, delta=delta, horizon=horizon, 
                                verbose=verbose)

  return LTL_Policy(proplist, dfsa.Tmat_csr, list(dfsa_init)[0], dfsa_final, pol, val)


class LTL_Policy(object):
  """control policy"""
  def __init__(self, proplist, dfsa_Tlist, dfsa_init, dfsa_final, pol, val):
    '''create a control policy object'''
    self.proplist = proplist
    self.dfsa_Tlist = dfsa_Tlist
    self.dfsa_init = dfsa_init
    self.dfsa_final = dfsa_final
    self.pol = pol
    self.val = val

    self.dfsa_state = self.dfsa_init

  def reset(self):
    '''reset controller'''
    self.dfsa_state = self.dfsa_init

  def report_aps(self, aps):
    '''report atomic propositions to update internal controller state'''
    dfsa_action = 0
    for x in map(lambda p: self.proplist.get(p, 0), aps):
      dfsa_action |= x

    row = self.dfsa_Tlist[dfsa_action].getrow(self.dfsa_state)
    assert row.nnz == 1

    self.dfsa_state = row.indices[0]

  def __call__(self, syst_state, t=0):
    '''get input from policy'''
    idx = tuple(syst_state) + (self.dfsa_state,)
    return self.pol[t][idx], self.val[t][idx]

  def finished(self):
    '''check if policy reached target'''
    return self.dfsa_state in self.dfsa_final
