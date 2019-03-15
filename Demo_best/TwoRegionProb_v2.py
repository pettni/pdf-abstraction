#
import random
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

import aux as rf
from fsa import Fsa
from hVI_fsrm import SPaths
from hVI_fsrm import Spec_Spaths
from hVI_fsrm import optimizers
from hVI_models import State_Space, Det_SI_Model
from hVI_types import Env

# %config InlineBackend.figure_format = 'retina'


print("Setting up Scenario")
sc = 'rss'  # Select Scenario 'toy' or 'rss'
obs_action = True  # Use separate action for observation
load = False  # Reads value function from pickle
parr = False  # Uses different threads to speed up value iteration
epsilon = 1e-10  # Used to compare floats while updating policy
rand_seed = 12

# Define Regions
# Regs have the same format as RSS code. Regs that are added first have a higher priority

# Wall region
print("Started wall case")
regs = OrderedDict()

a1 = rf.vertex_to_poly(np.array([[4, 0], [5, 0], [5, 1], [4, 1]]))
regs['a1'] = (a1, 0.3, 'sample1', 0)

a2 = rf.vertex_to_poly(np.array([[4, -3], [5, -3], [5, -2], [4, -2]]))
regs['a2'] = (a2, 0.6, 'sample3', 1)


output_color = {'a1': 'green', 'a2': 'blue', 'null': 'white'}

# Define Null regions with bounds of the space for null output with lowest priority
p = rf.vertex_to_poly(np.array([[-3, -5], [-3, 5], [5, -5], [5, 5]]))
regs['null'] = (p, 1.0, 'null')

# Construct belief-MDP for env
env = Env(regs)

''' Configuration Parameters '''
random.seed(rand_seed)
np.random.seed(rand_seed)

print('''--- Setup Motion and Observation Models ---''')
# Define Motion and Observation Model
Wx = np.eye(2)
Wu = np.eye(2)
r2_bs = State_Space([-5, -5], [5, 5])
motion_model = Det_SI_Model(0.1)

print("---- Constructing ROADMAP ----")
fig = plt.figure(1)
ax = fig.add_subplot(111, aspect='equal')
prm = SPaths(r2_bs, motion_model, Wx, Wu, regs, output_color, ax)
prm.make_nodes_edges(7, 3, means=[np.array([[-4.5], [0]]), np.array([[-2], [0]]), np.array([[0], [0]]),
                                  np.array([[2], [0]]), np.array([[4.5], [-.5]]),
                                  np.array([[4.5], [0.5]]), np.array([[4.5], [-2.5]])])

prm.plot(ax)
from matplotlib2tikz import save as tikz_save

# tikz_save("PRM1.tex")
plt.show()


print('-- Generate the DFA and the Product model----')

props = ['obs', 'sample1', 'sample3']
props = dict(zip(props, map(lambda x: 2 ** x, range(0, len(props)))))


fsaform = Fsa()
form = '! obs U ( sample1 | sample3 )'
#form = '( ! obs U sample3 ) & ( ! sample3 U sample2 )'

fsaform.from_formula(form)
formula_fsa = dict()
formula_fsa['fsa'] = fsaform
formula_fsa['init'] = dict({0: 1})
formula_fsa['final'] = {1}
formula_fsa['prop'] = props

print('--- Construct product PRM Spec_Spaths---')

prod_ = Spec_Spaths(prm, form, env, n=1,b_dist='U')

print('--- Start Back-ups ---')

not_converged = True
n = prod_.init[0]

opts = dict()
fig_nodes = len(list(key for key in prod_.active if prod_.active[key]==True))
for i in range(10):

    print('iteration', i)
    import hVI_algrthm

    hVI_algrthm.BP_local(prod_, 70)
    not_converged = prod_.full_back_up(opts)
    opt = np.unique(prod_.val[n].best_edge)
    j = 0
fig = plt.figure(figsize=(14, 30), dpi=120, facecolor='w', edgecolor='k')

print(prod_.sequence)
for n1 in prod_.active:
    if prod_.active[n1] == False:
        continue
    plt.subplot(fig_nodes, 2, j+1)
    prod_.plot_bel(n1, 0)
    plt.subplot(fig_nodes, 2, j+2)
    prod_.plot_bel(n1, 1)
    j = j + 2
    print(n1)
    plt.title(str(n1))

# tikz_save("PRM2.tex")

plt.show()



# # Prune
fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')
prm.plot(ax)
nodes, edges, visited = optimizers(prod_, ax)
# prod_.prune(keep_list=visited)

fig = plt.figure()

# recompute prod_.b_reg_set
b_reg_set = []
for b in prod_.b_prod_set:

    bi = np.reshape(b,(2,2))
    io = np.sum(bi,axis=0)
    i1 = np.sum(bi,axis=1)
    print(io[0, 1])
    b_reg_set += [np.matrix([[io[0,1]],[i1[1,0]]])]


print('breg',b_reg_set)

plt.scatter([i[0] for i in b_reg_set], [i[1] for i in b_reg_set])
# tikz_save("belief.tex")
plt.show()

import aux
fig=plt.figure()
v_list, v_2,d_2, act_list,obs_list, b_list,dist_list= aux.simulate(prod_,regs, fig=fig)
plt.show()