import polytope as pc
import cvxpy as cvx
import numpy as np
from numpy.linalg import inv
import scipy.optimize

def eps_err(lti,Dist,lamb=.99999):
  """
  Quantify accuracy of simulation with respect to disturbance given as a polytope
  :param lti: contains dynamics matrix lti.a, lti.b
  :param Dist: The disturbance given as a polytope
  :return: Invariant set R and epsilon
  """
  n = lti.dim
  m = lti.m
  A = lti.a
  B = lti.b
  C = lti.c

  Vertices = pc.extreme(Dist)

  # Introduce variables
  Minv = cvx.Variable((n, n), PSD=True)
  L    = cvx.Variable((m,n))
  eps2 = cvx.Variable((1, 1), nonneg=True)
  lam  = cvx.Parameter(nonneg=True, value=lamb)

  basic = cvx.bmat([[cvx.diag(np.ones((n,1))*lam) * Minv, np.zeros((n,1)),  Minv * A.T + L.T * B.T ],
                    [np.zeros((1,n)), (1-lam) * eps2, np.zeros((1,n))],
                    [A * Minv + B * L , np.zeros((n,1)), Minv]])

  cmat = cvx.bmat([[Minv, Minv * C.T],[C* Minv, np.eye(C.shape[0])]])
  constraintstup = (cmat >> 0,)

  ri =  np.zeros((n,1))
  for i in range(Vertices.shape[0]):
    ri = Vertices[i].reshape((n,1))
    rmat = cvx.bmat([[np.zeros((n, n)), np.zeros((n, 1)), np.zeros((n, n))],
                     [np.zeros((1, n)), np.zeros((1, 1)), ri.T],
                     [np.zeros((n, n)), ri, np.zeros((n, n))] ]   )
    constraintstup += (basic + rmat >> 0,)
  constraints = list(constraintstup)

  obj = cvx.Minimize(eps2)
  prob = cvx.Problem(obj, constraints)

  def f_opt(val):
    lam.value = val
    try:
      prob.solve()
    except cvx.error.SolverError :
      return np.inf

    return eps2.value[0,0]**.5

  lam_opt = scipy.optimize.fminbound(lambda val: f_opt(val), 0,1)
  lam.value = lam_opt
  prob.solve()
  eps_min = eps2.value[0,0] ** .5
  M_min = inv(Minv.value)
  K_min = L.value*Minv.value

  print ("status:", prob.status)
  print ("optimal epsilon", eps_min)
  print ("optimal M", M_min)
  print ("Optimal K", K_min)

  return M_min, K_min, eps_min