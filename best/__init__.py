# __all__ = ["abstraction", "mdp", "fsa", "ltl"]

from itertools import combinations
from operator import mul
import numpy as np

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

def prod(l):
  '''compute product of all elements in list l'''
  return reduce(mul, l, 1)

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
	'''multiplication of sparse matrix a with tensor b along dimension dim:
	   c(i1, ...ij, ..., in) = \sum_{ij'} a(ij,ij') b(i1, ..., ij', ..., in)'''

	assert a.shape[1] == b.shape[dim]

	new_shape = list(b.shape)
	new_shape[dim] = a.shape[0]

	return fold(a.dot(unfold(b, dim)), dim, new_shape)

def idx_to_midx(idx, n_list):
  '''index k to multiindex (i1, \ldots, in)   where i0 <= j < n_list[j]'''
  assert idx >= 0
  assert idx < prod(n_list)

  return tuple(idx % prod(n_list[i:]) / prod(n_list[i + 1:]) for i in range(len(n_list)))

def midx_to_idx(midx, n_list):
  '''multiindex (i1, \ldots, in) to index k'''
  assert len(midx) == len(n_list)
  assert all(midx[i] < n_list[i] for i in range(len(midx)))
  assert all(midx[i] >= 0 for i in range(len(midx)))

  return sum(midx[i] * prod(n_list[i+1:]) for i in range(len(n_list)))
