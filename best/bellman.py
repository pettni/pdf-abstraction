import numpy as np
import scipy.sparse as sp

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

    trans_qqa = np.zeros((dfa.N,dfa.N,len(dfa.Tmat_csr))) # q,q',act
    trans_qs = np.zeros((dfa.N,dfa.N, mdp.N)) # q, q',S'

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
    return V,pol, W