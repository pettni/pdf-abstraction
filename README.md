# pdf-abstraction

## Installation

This package is developed for Python 2.7. Install it as follows:

	python setup.py install

For development (create links to source code, no need to reinstall after making changes):

	python setup.py develop

Run tests:

	nosetests

## List of TODOs

 1. Move everything into ```best/``` and delete superfluous code
    - Move ```Linear.py``` to ```best/Linear.py```
    - Get rid of dependence on ```Models/MDP.py```
    - Integrate ```LTI_simrel``` into a new file ```best/simrel.py```
    - Get rid of dependence on ```pymdptoolbox```

 2. Write tests for abstractions simulation relations

 3. Implement uniform treatment of regions and abstractions via predicates

 4. Add LTL converter for Buchi automata

## Workflow

Given a specification written as a syntactically co-safe LTL property and a POMDP model which is
- a finite state partially observable Markov decision process, or
- a LTI system with gaussian disturbances
compute a policy with guaranteed probability that the property is satisfied.


1. Convert syntactically co-safe LTL property to DFA
  * (code manually until cristi is ready)

2. Formal abstraction of POMDP to approximate belief space model
*output:*
  * MDP (Markov decision process)
  * \delta = probabilistic deviation
  * Lset = set valued labeling map
  * R^{-1}(x) = set valued function containing all abstract states related to concrete state

3. Compute delta-robust game over cross product
*output:*
  * Value function for abstract model
  * Policy for abstract model

4. Implementation
  * refine policy  based on Value function, relation, and abstract Policy
  * simulate refined policy with concrete POMDP
