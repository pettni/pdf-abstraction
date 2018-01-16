# __all__ = ["abstraction", "mdp", "fsa", "ltl"]

from itertools import combinations
from operator import mul
import tensorly as tl

def subsets(collection):
    for i in range(len(collection)+1):
        it = combinations(collection, i)
        try:
            while True:
                yield(list(next(it)))
        except StopIteration:
            pass
    raise StopIteration

def prod(n):
  return reduce(mul, n, 1)

def sparse_tensordot(a, b, dim):
	# multiplication of sparse matrix a with tensor b along dimension dim:
	# c(i1, ...ij, ..., in) = \sum_{ij'} a(ij,ij') b(i1, ..., ij', ..., in)

	assert a.shape[1] == b.shape[dim]

	new_shape = list(b.shape)
	new_shape[dim] = a.shape[0]

	return tl.fold(a.dot(tl.unfold(b, dim)), dim, new_shape)