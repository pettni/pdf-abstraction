import numpy as np
from itertools import product

def sparse_tensor(a, b, dim):
	# multiplication of sparse matrix a with tensor b along dimension dim:
	# c(i1, ...ij, ..., in) = \sum_{ij'} a(ij,ij') b(i1, ..., ij', ..., in)

	assert a.shape[1] == b.shape[dim]

	new_shape = list(b.shape)
	new_shape[dim] = a.shape[0]

	c = np.zeros(new_shape)

	for midx_before in product(*[range(i) for i in b.shape[:dim]]):
		for midx_after in product(*[range(i) for i in b.shape[dim+1:]]):
			slicer = midx_before + (slice(None, a.shape[1], None),) + midx_after
			c[slicer] = a.dot(b[slicer])

	return c