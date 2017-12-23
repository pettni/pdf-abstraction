# __all__ = ["abstraction", "mdp", "fsa", "ltl"]

from itertools import combinations
from operator import mul

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
