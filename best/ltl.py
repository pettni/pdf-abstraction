import numpy as np
from itertools import product

from best.fsa import Fsa
from best.policy import LTL_Policy
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


def solve_ltl_cosafe(mdp, formula, connection, maxiter=np.Inf):
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

  dfsa, init, final, _ = formula_to_mdp(formula)
  prod = ProductMDP(mdp, dfsa, connection)
  V, pol = prod.solve_reach(accept=lambda s: s[1] in final, maxiter=maxiter)

  pol_qn = pol.reshape( (dfsa.N, mdp.N), order='F' )
  V_qn = V.reshape( (dfsa.N, mdp.N), order='F' )

  return LTL_Policy(dfsa, list(init)[0], final, pol_qn, V_qn)

