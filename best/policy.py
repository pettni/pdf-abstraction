import numpy as np
import scipy.sparse as sp


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


class RefinedPolicy(): # refined policy

  def __init__(self, abstr, W, ltlpolicy):
    self.abstr = abstr
    self.ltlpolicy = ltlpolicy
    
    self.W = W
    
    # interface
    self.K = np.zeros((len(abstr.urep), len(abstr.srep)))

  def __call__(self, s_concrete):
    """
    :param s_next:  s_{k+1}
    :return: input
    """

    # get APs at concrete state
    aps = self.abstr.aps(s_concrete)

    # update policy state
    self.ltlpolicy.report_aps(aps)

    # get next abstract MDP state 
    # TODO: choose maximizer for V?
    s_abstract = self.abstr.closest_abstract(s_concrete)   

    # compute abstract input
    u_abstract, curr_val = self.ltlpolicy.get_input(s_abstract)
    
    # refine input
    return self.abstr.refine_input(u_abstract, s_abstract, s_concrete)

  def reset(self):
    self.ltlpolicy.reset()

  def finished(self):
    return self.ltlpolicy.finished()

  def cst(self, s_concrete):
    """
    :param s_next:  s_{k+1}
    :return: input
    """

    if s_concrete.shape[1]>1:
      u = np.zeros((len(self.abstr.urep),s_concrete.shape[1] ))
      for i in range(s_concrete.shape[1]):
        u[:,i] = self.cst( s_concrete[:,[i]]).flatten()
      return u

    u = self.__call__(s_concrete)
    self.reset()
    return u

  def val_concrete(self, s_concrete):
    """
    :param s_concrete: before transformation state
    :return:
    """
    if s_concrete.shape[1] > 1:
      val = np.zeros((s_concrete.shape[1],))
      for i in range(s_concrete.shape[1]):
        valdd = self.val_concrete(s_concrete[:, i].reshape((-1, 1))).flatten()
        try :
          val[i] = valdd
        except :
          val[i] = 0
          print('ERR')

      return val

    s_abstract = self.abstr.closest_abstract(s_concrete)   

    if s_abstract < self.abstr.mdp.N-1:
      val = self.W[s_abstract]
    else:
      val = 0

    return np.array([[val]])
