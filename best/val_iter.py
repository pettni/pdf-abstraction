import time
import numpy as np
from best import DTYPE


def solve_reach(network, accept, horizon=np.Inf, delta=0, prec=1e-5, verbose=False):
  '''solve reachability problem
  Inputs:
  - accept: function defining target set
  - horizon: reachability horizon (standard is infinite-horizon)
  - delta: failure probability in each step
  - prec: termination tolerance (inifinite-horizon case)

  Outputs::
  - val_list: array [V0 V1 .. VT] of value functions
  - pol_list: array [P0 P1 .. PT] of value functions

  The infinite-horizon problem has a stationary value function and policy. In this
  case the return arguments have length 1, i.e. val_list = [V0], pol_list = [P0].
  '''

  V_accept = accept.astype(DTYPE, copy=False)

  it = 0
  start = time.time()

  V = np.fmax(V_accept, np.zeros(network.N, dtype=DTYPE))

  val_list = []
  pol_list = []
  val_list.insert(0, V)

  slice_names = network.state_names
  assert(len(slice_names) == len(V.shape))

  while it < horizon:

    V_new = V

    if verbose:
      print('iteration {}, time {}'.format(it, time.time()-start))

    # Iterate bottom up 
    for pomdp in network.bottom_up_iter():

      # Do backup over current state
      if verbose:
        print("slices {}, bellman backup over {}".format(slice_names, pomdp.state_name))
      
      V_new = pomdp.bellman_(V_new, slice_names.index(pomdp.state_name))

      print(V_new)

      slice_names = pomdp.input_names + slice_names
      assert(len(slice_names) == len(V_new.shape))

      # Resolve connections (get rid of unfree actions)
      for _, _, attr in network.graph.in_edges(pomdp, data=True):
        if attr['input'] in slice_names:

          if verbose: 
            print('slices {}, resolving {} -> {}'
                  .format(slice_names, attr['output'], attr['input']))

          dim_x = slice_names.index(attr['output'])
          dim_u = slice_names.index(attr['input'])
          conn_mat = attr['conn_mat'].transpose()

          # reshape to same dim as V_new
          new_shape = np.ones(len(V_new.shape), dtype=np.uint32)
          new_shape[dim_u] = conn_mat.shape[0]
          new_shape[dim_x] = conn_mat.shape[1]

          conn_mat = conn_mat.reshape(new_shape)
          print(new_shape)
          print('connmat', conn_mat)

          V_new = np.maximum(1-conn_mat, V_new).min(axis=dim_u)

          slice_names = tuple(name for name in slice_names if name != attr['input'])
          assert(len(slice_names) == len(V_new.shape))

      print(V_new)

      # Max over free actions
      free_actions = (name for name in pomdp.input_names if name in network.input_names)

      for free_action in free_actions:

        if verbose:
          print("slices {}, max over {}".format(slice_names, free_action))
        
        V_new = V_new.max(axis=slice_names.index(free_action))
        
        slice_names = tuple(name for name in slice_names if name != free_action)
        assert(len(slice_names) == len(V_new.shape))

      print(V_new)

    # Max over accept set
    V_new = np.fmax(V_accept, np.maximum(V_new - delta, 0))

    if horizon < np.Inf and it < horizon-1:
      val_list.insert(0, V_new)
      pol_list.insert(0, None)

    if horizon == np.Inf and np.amax(np.abs(V_new - V)) < prec:
      break
    V = V_new

    it += 1

  val_list.insert(0, V_new)
  pol_list.insert(0, None)

  print('finished after {}s and {} iterations'.format(time.time()-start, it))

  return val_list, pol_list
