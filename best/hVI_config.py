''' Configuration Parameters '''
sc = 'rss'  # Select Scenario 'toy' or 'rss'
obs_action = True  # Use seperate action for observation
load = False  # Reads value function from pickle
parr = False   # Uses different threads to speed up value iteration
epsilon = 1e-10  # Used to compare floats while updating policy
rand_seed = 12
