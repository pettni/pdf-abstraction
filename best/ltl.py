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

  mdp.init = list(init_states)

  # create map number => sets of atom propositions
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

  return mdp, init_states, final_states, dict_input2prop


def solve_ltl_cosafe(mdp, formula, connection, maxiter=np.Inf, delta=0., verbose=False):
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

  dfsa, dfsa_init, dfsa_final, _ = formula_to_mdp(formula)

  prod_mdp = mdp.product(dfsa, connection)

  Vacc = np.zeros(prod_mdp.N_list)
  Vacc[...,list(dfsa_final)[0]] = 1

  V, pol = prod_mdp.solve_reach(Vacc, delta=delta, maxiter=maxiter, 
                                verbose=verbose)

  pol_qn = pol.ravel().reshape( (dfsa.N, mdp.N), order='F' )
  V_qn = V.ravel().reshape( (dfsa.N, mdp.N), order='F' )

  return LTL_Policy(dfsa, list(dfsa_init)[0], dfsa_final, pol_qn, V_qn)


class LTL_Policy(object):
  """control policy"""
  def __init__(self, dfsa, dfsa_init, dfsa_final, pol, V):
    '''create a control policy object'''
    self.dfsa = dfsa
    self.dfsa_init = dfsa_init
    self.dfsa_final = dfsa_final
    self.pol = pol
    self.V = V

    self.dfsa_state = self.dfsa_init

  def reset(self):
    '''reset controller'''
    self.dfsa_state = self.dfsa_init

  def report_aps(self, aps):
    '''report atomic propositions to update internal controller state'''

    # todo: get rid of stupidity (use csr matrix)
    dfsa_action = self.dfsa.input_fcn( aps )

    dfsa_mdp_state = np.zeros((self.dfsa.N, 1))
    dfsa_mdp_state[self.dfsa_state] = 1
    dfsa_mdp_state = self.dfsa.evolve(dfsa_mdp_state, dfsa_action)
    
    self.dfsa_state = np.argmax(dfsa_mdp_state)

  def get_input(self, syst_state):
    '''get input from policy'''
    return self.pol[self.dfsa_state, syst_state], self.V[self.dfsa_state, syst_state]

  def finished(self):
    '''check if policy reached target'''
    return self.dfsa_state in self.dfsa_final
