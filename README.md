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

## POMDP branch

### Accomplished

 - Novel POMDP class
 - New connection structure (graph instead of recursive parallel/serial)
 - Switched to Python 3

### TODO

 - Convert old examples (lti, ADHS, RSS18) to new format
 - Base class for policies, solvers return subclasses
 - Competing types of value iteration: 
    * [x] sparse tables (for MDPs)
    * [ ] mtBDD? (for MDPs)
    * [ ] VDC
    * [ ] DQN via openai.baseline
    * [ ] pbVI by converting Rohan's code?


