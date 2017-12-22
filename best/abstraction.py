import numpy as np
import polytope as pc
import itertools
import operator
from scipy.stats import norm

import matplotlib.pyplot as plt

from best.mdp import MDP
from ApprxSimulation.LTI_simrel import eps_err

def prod(n):
  return reduce(operator.mul, n, 1)

class LTIAbstraction(object):

  def __init__(self, lti_syst_orig, d, un=3, verbose = True, Accuracy=True):

    self.s_finite = None

    # Diagonalize
    lti_syst = lti_syst_orig.normalize()
    self.T2x = lti_syst.T2x

    ## Unpack LTI
    d = d.flatten()
    A = lti_syst.a
    B = lti_syst.b
    C = lti_syst.c
    U = lti_syst.setU()

    # check that Bw is a diagonal
    assert np.sum(np.absolute(lti_syst.W)) - np.trace(np.absolute(lti_syst.W)) == 0
    vars = np.diag(lti_syst.W)

    X = lti_syst.setX()
    n = lti_syst.dim

    rad = np.linalg.norm(d, 2)
    lx, ux = pc.bounding_box(lti_syst.X)  # lower and upperbounds over all dimensions
    remainx = np.remainder((ux-lx).flatten(),d.flatten())
    remainx = np.array([d.flatten()[i]-r if r!=0 else 0 for i,r in enumerate(remainx) ]).flatten()
    lx =lx.flatten() - remainx/2
    ux =ux.flatten() + d

    if Accuracy:
      Dist = pc.box2poly(np.diag(d).dot(np.kron(np.ones((lti_syst.dim, 1)), np.array([[-1, 1]]))))
      M_min, K_min, eps_min = eps_err(lti_syst, Dist)
    else:
      M_min, K_min, eps_min = None, None, None

    # grid state
    srep = tuple()
    sedge = tuple()
    for i, dval in enumerate(d):
      srep += (np.arange(lx[i], ux[i], dval)+dval/2,)
      sedge += (np.arange(lx[i], ux[i]+dval, dval),)

    sn_list = [len(sr) for sr in srep]
    sn = prod(sn_list) # number of finite states

    # grid input
    urep = tuple()
    lu, uu = pc.bounding_box(lti_syst.U)  # lower and upperbounds over all dimensions
    for i, low in enumerate(lu):
      urep += (np.linspace(lu[i], uu[i], un, endpoint=True),)

    un_list = [len(ur) for ur in urep]
    un = prod(un_list)  # number of finite states

    transition_list = [np.zeros((sn+1, sn+1)) for m in range(un)]

    # extract all transitions
    for u_index, u in enumerate(itertools.product(*urep)):
      P = tuple()
      for s, sstate in enumerate(itertools.product(*srep)):
        mean = np.dot(A, np.array(sstate).reshape(-1, 1)) + np.dot(B, np.array(u).reshape(-1, 1))  # Ax

        # compute probability in each dimension
        Pi = tuple()
        for i in range(n):
          if vars[i]>np.finfo(np.float32).eps:
            Pi += (np.diff(norm.cdf(sedge[i], mean[i], vars[i] ** .5)).reshape(-1),)  # probabilities for different dimensions
          else:
            abs_dis = np.array(map(lambda s: abs(s - mean[i]), srep[i]))
            p_loc = np.zeros(srep[i].shape)
            p_loc[abs_dis.argmin()] = 1
            Pi += (p_loc,)

        # multiply over dimensions
        P += (np.array([[reduce(operator.mul, p, 1) for p in itertools.product(*Pi)]]),)

      prob = np.concatenate(P, axis = 0)
      p_local = np.block([[prob, 1-prob.dot(np.ones((prob.shape[1], 1)))], [np.zeros((1, prob.shape[1])), np.ones((1,1))]])

      transition_list[u_index] = p_local


    # x-valued output
    # mapping index -> center point
    def output_fcn(s):
      if s == self.mdp.N-1:
        return (s, (None,) * len(sn_list))
      else:
        x_diag = np.array(tuple(srep[i][s % prod(sn_list[i:]) / prod(sn_list[i + 1:])]
                          for i in range(len(sn_list)))).reshape(2,1)
        return (s,  self.transform_d_o(x_diag))

    self.mdp = MDP(transition_list, input_name='u_d', output_name='(s, x_d)', 
                   output_fcn=output_fcn)

    self.srep = srep
    self.urep = urep
    self.sedge = sedge

    self.M = M_min
    self.K = K_min
    self.eps = eps_min

    self.K_refine = np.zeros((len(self.urep), len(self.srep)))

    self.output_cst = dict([(s, sstate) for s, sstate in enumerate(itertools.product(*srep))])
    self.input_cst = dict([(u, uvalue) for u, uvalue in enumerate(itertools.product(*urep))])


  def set_regions(self, regions):
    self.ap_regions = regions


  def plot(self, fig):
    # todo: add transformation
    ax = fig.add_subplot(111)

    grid = np.meshgrid(*self.srep)

    ax.scatter(grid[0].flatten(), grid[1].flatten(), label='Finite states', color='k', s=10, marker="o")


  def closest_abstract(self, x):
    '''compute abstract state s closest to x and in simulation'''
    if self.s_finite is None:
      self.s_finite = np.array(list(itertools.product(*self.srep)))  # compute the grid points

    # convert to diagonal system
    x_diag = self.transform_o_d(x)

    # find closest abstract state
    sdiff = self.s_finite-np.tile(x_diag.reshape((1,-1)), (self.s_finite.shape[0], 1))
    error=np.diag(sdiff.dot(self.M).dot(sdiff.T))

    s = error.argmin()

    # no state simulates concrete state
    if error[s] >= self.eps**2:
      s = self.mdp.N-1

    return s

  def refine_input(self, u_ab, s_ab, s_conc):
    '''refine abstract input u_ab to concrete input'''
    u = np.array(self.input_cst[u_ab]).reshape(-1, 1)

    # give zero if in dummy state
    if s_ab == self.mdp.N - 1:
      u = np.zeros((len(self.input_cst[0]),1))

    u = self.K_refine.dot(s_conc-s_ab) + u
    return u # input

  def all_abstract(self, x):
    '''compute abstract states that are related to x via simulation relation
       - returns indices, points'''
    if self.s_finite is None:
      self.s_finite = np.array(list(itertools.product(*self.srep)))  # compute the grid points

    x_diag = self.transform_o_d(x)

    if self.M is None:
      print("WARNING no M matrix given")
      self.M =np.eye(len(self.srep))

    if self.eps is None:
      print("WARNING no epsilon give")
      self.eps = 1

    # quantify the weighted difference between x_diag, and values of s
    sdiff = self.s_finite-np.tile(x_diag.reshape((1,-1)), (self.s_finite.shape[0], 1))
    error=np.diag(sdiff.dot(self.M).dot(sdiff.T))
    s_range = np.arange(self.mdp.N-1) # minus to remove dummy state
    return s_range[error<=self.eps**2], self.s_finite[error<=self.eps**2]

  def aps(self, x):
    '''return atomic propositions at concrete state x (original coordinates)'''
    aps = tuple()
    for input_i in self.ap_regions.keys():
      if self.ap_regions[input_i].contains(x):
        aps += (input_i,)
    return aps

  def transform_o_d(self, x):
    '''transform from original to diagonal coordinates'''
    return np.linalg.inv(self.T2x).dot(x)

  def transform_d_o(self, x_diag):
    '''transform from diagonal to original coordinates'''
    return self.T2x.dot(x_diag)

  def map_dfa_inputs(self):
    '''for dict regions {'region': polytope}, compute dicts in_regions and nin_regions where
      in_regions['region'][s] = 1 if abstract state s may be in region and 0 otherwise
      in_regions['region'][s] = 1 if abstract state s may be outside region and 0 otherwise '''

    in_regions = dict()
    nin_regions = dict()

    if (self.eps is None) | (self.eps ==0):
      for input_i in self.ap_regions.keys():
        in_regions[input_i] = np.array([[1.] if self.T2x.dot(np.array(s)) in self.ap_regions[input_i] else [0.] for s in itertools.product(*self.srep)])

      for input_i in self.ap_regions.keys():
        nin_regions[input_i] = np.ones(in_regions[input_i].shape)-np.array([[1.] if self.T2x.dot(np.array(s)) in regions[input_i] else [0.] for s in itertools.product(*self.srep)])

    else :
      u, s, v = np.linalg.svd(self.M)
      Minvhalf = np.linalg.inv(v).dot(np.diag(np.power(s, -.5)))
      Minhalf = np.diag(np.power(s, .5)).dot(v)
      # eps is not =0,
      for input_i in self.ap_regions.keys(): # for each region, which is a polytope. Check whether it can be in it
        #Big_polytope = regions[input_i] #<--- decrease size polytope

        A = self.ap_regions[input_i].A.dot(self.T2x).dot(Minvhalf)
        b = self.ap_regions[input_i].b

        scaling = np.zeros((A.shape[0],A.shape[0]))
        for index in range(A.shape[0]):
            scaling[index,index] = np.linalg.norm(A[index,:])**-1
        print('check norm of rows', scaling.dot(A))
        A = scaling.dot(A).dot(Minhalf)

        b = scaling.dot(b) + self.eps

        assert self.ap_regions[input_i].A.shape == A.shape
        assert self.ap_regions[input_i].b.shape == b.shape

        in_regions[input_i] = np.array(
            [[1.] if np.all(A.dot(np.array(s))-b<=0) else [0.] for s in
             itertools.product(*self.srep)])

      for input_i in self.ap_regions.keys():
        # quantify whether a state could be outside the polytope
        A = self.ap_regions[input_i].A.dot(self.T2x).dot(Minvhalf)
        b = self.ap_regions[input_i].b

        scaling = np.zeros((A.shape[0],A.shape[0]))
        for index in range(A.shape[0]):
            scaling[index,index] = np.linalg.norm(A[index,:])**-1

        A = scaling.dot(A).dot(Minhalf)
        b = scaling.dot(b) - self.eps

        nin_regions[input_i] = np.ones(in_regions[input_i].shape) - np.array(
            [[1.] if np.all(A.dot(np.array(s))-b<=0)else [0.] for s in
             itertools.product(*self.srep)])
    
    return in_regions, nin_regions
