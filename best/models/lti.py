import numpy as np
import itertools
import polytope as pc
from numpy import linalg as LA
import matplotlib.pyplot as plt

from scipy.stats import norm
from functools import reduce # Valid in Python 2.6+, required in Python 3

class LTI:
  """Define a discrete-time linear time invariant system"""

  def __init__(self, a, b, c, d, x=None, bw=None, W=None, u=None, T2x = None,stochdiff=None):
    self.a = a
    self.b = b
    self.c = c  # currently unused
    self.d = d  # currently unused
    self.dim = len(a)
    self.m = b.shape[1]
    self.X = x
    self.T2x = T2x

    self.stochdiff = stochdiff
    if W is None:
      if bw is None:
        self.W = None
        self.bw = None
      else:
        self.bw = bw
        self.W = bw.dot(bw.T)
    else:
      self.bw = None
      self.W = W

    self.U = u

  def setU(self, u=None):
    if isinstance(u,pc.Polytope):
      self.U = u
      return self.U
    if u is None:
      print('Warning no inputspace given')
      if self.U is None:
        print('Define standard box polytope 0-1')
        self.U = pc.box2poly(np.kron(np.ones((self.m, 1)), [-1, 1]))
        return self.U
      else:
        return self.U

  def setX(self, x=None):
    if isinstance(x,pc.Polytope):
      self.X = x
      return self.X
    else:
      print('Warning no state space given')
      if self.X is None:
        print('Define standard box polytope -1,1')
        self.X = pc.box2poly(np.kron(np.ones((self.dim, 1)), np.array([[-1, 1]])))
        return self.X
      else:
        return self.X

  def setBw(self, bw=None):
    if isinstance(bw,np.ndarray) :
      self.bw = bw
      self.W = bw.dot(bw.T)

      return self.bw
    if bw is None:
      print('Warning no matrix BW given')
      if self.bw is None:
        print('Define matrix Bw')
        self.bw = np.eye(self.dim)
        self.W = self.bw.dot(self.bw.T)

        return self.bw
      else:
        print('keep matrix Bw')
        return self.bw

  def normalize(self):
    # compute svd
    # compute singular value decomposition
    # Meps = U*s*V, Meps**.5=U*s**.5
    U, s, V = np.linalg.svd(self.W, full_matrices=True)

    #x_trans = U.T * x
    a_trans = U.T.dot(self.a).dot(U)
    b_trans = U.T.dot(self.b)
    c_trans = self.c.dot(U)
    d_trans = self.d

    # product over polytope
    X_trans = pc.Polytope(A = self.X.A.dot(U), b = self.X.b, normalize = False)
    sys_n = LTI(a_trans, b_trans, c_trans, d_trans, x=X_trans,u=self.U, W = np.diag(s), T2x = U)

    return sys_n
