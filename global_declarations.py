#!/usr/bin/env python
"""Short title.

Long explanation
"""


# List of info
__author__ = "Sofie Haesaert"
__copyright__ = "Copyright 2018, TU/e"
__credits__ = ["Sofie Haesaert"]
__license__ = "BSD3"
__version__ = "1.0"
__maintainer__ = "Sofie Haesaert"
__email__ = "s.haesaert@tue.nl"
__status__ = "Draft"

#''' Configuration Parameters '''
sc = 'rss'  # Select Scenario 'toy' or 'rss'
obs_action = True  # Use separate action for observation
load = False  # Reads value function from pickle
parr = False   # Uses different threads to speed up value iteration
epsilon = 1e-10  # Used to compare floats while updating policy
rand_seed = 12