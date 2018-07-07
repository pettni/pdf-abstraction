# pdf-abstraction

## Installation

Install necessary packages

  sudo apt install graphviz-dev 
  pip install -r requirements.txt

This package is developed for Python 2.7. Install it as follows:

  python setup.py install

For development (create links to source code, no need to reinstall after making changes):

  python setup.py develop

Run tests:

  nosetests

## TODO pomdp branch

 - Competing types of value iteration: 
    * [x] sparse tables (for MDPs)
    * [ ] mtBDD? (for MDPs)
    * [ ] VDC
    * [ ] DQN
    * [ ] pbVI?
 - Allow many-to-one connections
 - Automatic attachment of LTL automaton given AP definitions
 - LTL policy
 - Interface to work with OpenAI baseline (gym environment)
 - Switch to Python 3
 - Separate out abstraction and firm code 


