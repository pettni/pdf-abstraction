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


def solve_ltl_cosafe(mdp, formula, connection, maxiter=np.Inf, delta=0.,
                     algorithm='sofie', verbose=False):
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

  if algorithm == 'petter':
    prod_mdp = mdp.product(dfsa, connection)
    V, pol = prod_mdp.solve_reach(accept=lambda s: s[-1] in dfsa_final, 
                                  delta=delta, maxiter=maxiter, 
                                  verbose=verbose)
  else:
    act_inputs = np.array([[1 if q in map(dfsa.input, connection( mdp.output(s) )) else 0 
                            for s in range(mdp.N)] 
                           for q in range(dfsa.M)])
    maxiter = min(maxiter, 100)  # todo: fix
    V, pol, _ = reach_dfa(act_inputs, mdp, dfsa, dfsa_final, 
                          recursions=maxiter, delta=delta)

  pol_qn = pol.reshape( (dfsa.N, mdp.N), order='F' )
  V_qn = V.reshape( (dfsa.N, mdp.N), order='F' )

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


def reach_dfa(act_inputs, mdp, dfa, dfa_final, V = None, recursions=1, delta=0 ):
    # TODO : Add delta
    assert dfa is not None
    assert dfa_final is not None
    if V is None:
        # this V should be a reference to the data rather than a copy
        V = np.zeros((dfa.N, mdp.N))

    assert V.shape in ((dfa.N, mdp.N),)

    # let q\in Q be states of DFA
    # let S \in SS be states of MDP

    # create indicator functions for final, Q\(final)

    accept = lambda q: map(lambda qi: int(qi in dfa_final), q)

    naccept= lambda q: map(lambda qi: 1-int(qi in dfa_final), q)
    Accept= np.kron(np.array(accept(range(dfa.N))).reshape((dfa.N,1)), np.ones((1,mdp.N)))
    nAccept = np.diag(np.array(naccept(range(dfa.N))))

    Tnew = sp.hstack(dfa.Tmat_csr).toarray()
    #print(Tnew.toarray())

    trans_qqa = np.zeros((dfa.N, dfa.N, dfa.M)) # q, q', act
    trans_qs  = np.zeros((dfa.N, dfa.N, mdp.N)) # q, q', S'

    for q in range(dfa.N):
        trans_qqa[q] = Tnew[q,:].reshape((dfa.N, -1), order="F")
        array = np.zeros((dfa.N, mdp.N))
        bool_array =(trans_qqa[q].dot(act_inputs) < 1)
        array[bool_array] = 1000.
        trans_qs[q] = array #np.array((array < 1), dtype=np._float)*1000.0 # penalise impossible transitions

    trans_qs[:,:,-1]=np.zeros((dfa.N,dfa.N)) # set dummy state equal to zero

    # [T[0] T[1] T[2] ... ]
    # given q, S
    # next SxAct -> prob(S')
    # W = 1accept(qnext) + 1_{not accept }(qnext)  V
    pol = np.zeros((dfa.N, mdp.N))
    V_new = np.zeros((dfa.N, mdp.N))
    for rec in range(recursions):

        for q in range(dfa.N):
            W = np.amin(Accept+ nAccept.dot(V) + trans_qs[q],axis =0 ) # 1 x S'
            W_a = np.block([[(mdp.T(a).dot(W.T)).transpose()] for a in range(mdp.M)])
            # W_a = np.block([[W.dot(mdp.T(a).todense().T)] for a in range(mdp.M)])
            if rec == recursions-1 : # at last step also comput the policy
                pol[q] = W_a.argmax(axis = 0)
            if delta == 0:
                V_new[q] = W_a.max(axis = 0)   #max_{s_action}[ s_action X S]
            else:
                V_new[q] = W_a.max(axis = 0) - delta  #max_{s_action}[ s_action X S]

            V_new[q] = np.clip(V_new[q],0,1) # limit to values between zero and one
            #print(q, V_new[q].sum(axis =0))
        V = V_new

    W = np.amin(Accept+ nAccept.dot(V)+trans_qs[dfa.init[0]],axis =0 )
    return V, pol.astype(np.int32), W