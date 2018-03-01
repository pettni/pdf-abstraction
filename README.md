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
    - Get rid of dependence on ```Models/MDP.py``` and delete it, this will remove dependence on ```pymdptoolbox```
    - Integrate ```LTI_simrel``` into a new file ```best/simrel.py```

 2. Write tests for abstractions simulation relations

 3. Implement uniform treatment of regions and abstractions via predicates

 4. Add LTL converter for Buchi automata
 
## New things that may be implemented in the future

 1. FIRM abstractions (Rohan)
 
 2. Serial and parallell products with MDPs and POMDPs. Sequential point-based value iteration for such products. (Rohan, Petter)
 
 3. Barrier-function based abstractions (Petter)
