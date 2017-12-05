# pdf-abstraction

Given a specification written as a syntactically co-safe LTL property and a POMDP model which is
- a finite state partially observable Markov decision process, or
- a LTI system with gaussian disturbances
compute a policy with guaranteed probability that the property is satisfied.


1. Convert syntactically co-safe LTL property to DFA

2. Formal abstraction of POMDP to approximate belief space model
*output:*
  * MDP (Markov decision process)
  * \delta = probabilistic deviation
  * Lset = set valued labeling map

3. compute delta-robust game over cross product
*output:*
  * Value function for abstract prismmodelchecker



4. implementation
