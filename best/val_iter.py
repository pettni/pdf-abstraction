'''Module for table value iteration'''

import time
import numpy as np
from best import DTYPE, DTYPE_ACTION

def network_bellman(network, W, slice_names, verbose=False):
  '''calculate Q function via one Bellman step
     Q(u_free, x) = \sum_x' T(x' | x, u_free) W(x')  '''

  # Iterate bottom up 
  for pomdp in network.bottom_up_iter():

    # Do backup over current state
    if verbose:
      print("slices {}, bellman backup over {}".format(slice_names, pomdp.state_name))
    
    W = pomdp.bellman_(W, slice_names.index(pomdp.state_name))
    slice_names = list(pomdp.input_names) + slice_names

    # Resolve connections (non-free actions)
    for _, _, attr in network.graph.in_edges(pomdp, data=True):
      if attr['input'] in slice_names:

        if verbose: 
          print('slices {}, resolving {} -> {}'
                .format(slice_names, attr['output'], attr['input']))

        dim_x = slice_names.index(attr['output'])
        dim_u = slice_names.index(attr['input'])
        conn_mat = attr['conn_mat'].transpose() # conn_mat is [x,u] but we need [u,x] 

        # reshape to same dim as W
        new_shape = np.ones(len(W.shape), dtype=np.uint32)
        new_shape[dim_u] = conn_mat.shape[0]
        new_shape[dim_x] = conn_mat.shape[1]

        conn_mat = conn_mat.reshape(new_shape)

        W = np.maximum(1-conn_mat, W).min(axis=dim_u)

        slice_names.remove(attr['input'])

  return W, slice_names


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
  - pol_list: array [P0 P1 .. PT] of policy functions

  The infinite-horizon problem has a stationary value function and policy. In this
  case the return arguments have length 1, i.e. val_list = [V0], pol_list = [P0].
  '''

  V_accept = accept.astype(DTYPE, copy=False)

  it = 0
  start = time.time()

  V = np.fmax(V_accept, np.zeros(network.N, dtype=DTYPE))

  val_list = []
  val_list.insert(0, V)

  slice_names = list(network.state_names)

  while it < horizon:

    if verbose:
      print('iteration {}, time {}'.format(it, time.time()-start))

    # Calculate Q(u_free, x)
    Q, slice_names = network_bellman(network, V, slice_names, verbose)

    # Max over free actions
    for free_action in network.input_names:    
      Q = Q.max(axis=slice_names.index(free_action))
      slice_names.remove(free_action)

    # Max over accept set
    V_new = np.fmax(V_accept, np.maximum(Q - delta, 0))

    if horizon < np.Inf and it < horizon-1:
      val_list.insert(0, V_new)

    if horizon == np.Inf and np.amax(np.abs(V_new - V)) < prec:
      break
    V = V_new

    it += 1

  val_list.insert(0, V_new)

  print('finished after {}s and {} iterations'.format(time.time()-start, it))

  return val_list


def get_input(network, x, W):
  '''compute argmax_{u_free} E[W(x') | x, u_free] '''

  slice_names = list(network.state_names)
  assert(len(slice_names) == len(W.shape))

  # TODO: speed up by doing Bellman for a single initial state
  Q, slice_names = network_bellman(network, W, slice_names)

  slice_obj = (Ellipsis,) + tuple(x)

  val = np.max(Q[slice_obj])
  u_flat = np.argmax(Q[slice_obj])

  return np.unravel_index(u_flat, network.M), val