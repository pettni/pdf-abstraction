from itertools import combinations
import numpy as np
# import tensorly as tl

def subsets(collection):
  '''construct iterator over all subsets in collection'''
  for i in range(len(collection)+1):
    it = combinations(collection, i)
    try:
      while True:
        yield(list(next(it)))
    except StopIteration:
      pass
  raise StopIteration

def multidim_list(n_list):
  '''create multidimensional list of size n_list with elements None'''
  ret = None
  for n in reversed(n_list):
    ret = [ret] * n
  return ret

def len_multidim_list(multidim_list):
  '''return dimensions of multidimensional list (assuming hypercube)'''
  mlist = multidim_list
  ret = []

  while type(mlist) is list:
    ret.append(len(mlist))
    mlist = mlist[0]

  return ret

def prod(l):
  '''compute product of all elements in list l'''
  ret = 1
  for el in l:
    ret *= el
  return ret

def fold(unfolded_tensor, mode, shape):
  '''Fold a tensor (taken from tensorly)'''
  full_shape = list(shape)
  mode_dim = full_shape.pop(mode)
  full_shape.insert(0, mode_dim)
  return np.moveaxis(np.reshape(unfolded_tensor, full_shape), 0, mode)

def unfold(tensor, mode):
  '''Unfold a tensor in mode (taken from tensorly)'''
  return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))

def sparse_tensordot(a, b, dim):
  # multiplication of sparse matrix a with tensor b along dimension dim:
  # c(i1, ...ij, ..., in) = \sum_{ij'} a(ij,ij') b(i1, ..., ij', ..., in)

  assert a.shape[1] == b.shape[dim]

  new_shape = list(b.shape)
  new_shape[dim] = a.shape[0]

  return fold(a.dot(unfold(b, dim)), dim, new_shape)

def idx_to_midx(idx, n_list):
  # index k to multiindex (i1, \ldots, in)   where i0 <= j < n_list[j]
  assert idx >= 0
  assert idx < prod(n_list)

  return tuple(idx % prod(n_list[i:]) / prod(n_list[i + 1:]) for i in range(len(n_list)))

def midx_to_idx(midx, n_list):
  # multiindex (i1, \ldots, in) to index k
  assert len(midx) == len(n_list)
  assert all(midx[i] < n_list[i] for i in range(len(midx)))
  assert all(midx[i] >= 0 for i in range(len(midx)))

  return sum(midx[i] * prod(n_list[i+1:]) for i in range(len(n_list)))
