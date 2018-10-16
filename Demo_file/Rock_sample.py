from collections import OrderedDict
from best.hVI_fsrm import SPaths
from best.hVI_fsrm import spec_Spaths
from best.hVI_models import State_Space, Det_SI_Model
from best.hVI_types import Env, Gamma
import best.rss18_functions as rf
from best.hVI_config import sc, load, parr, obs_action, epsilon, rand_seed
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import random
import time


print "Setting Up Scenario"

# Define Regions
# Regs have the same format as RSS code. Regs that are added first have a higher priority
#### Wall region
print("Started wall case")
regs = OrderedDict()
p1 = rf.vertex_to_poly(np.array([[2, 0], [3, 0], [3, 1], [2, 1]]))
regs['r1'] = (p1, 1, 'obs')
p2 = rf.vertex_to_poly(np.array([[2, 1], [3, 1], [3, 2], [2, 2]]))
regs['r2'] = (p2, 1, 'obs')
p3 = rf.vertex_to_poly(np.array([[2, 2], [3, 2], [3, 3], [2, 3]]))
regs['r3'] = (p3, 1, 'obs')
p4 = rf.vertex_to_poly(np.array([[2, 3], [3, 3], [3, 4], [2, 4]]))
regs['r4'] = (p4, .6, 'obs', 1)
p5 = rf.vertex_to_poly(np.array([[2, 4], [3, 4], [3, 5], [2, 5]]))
regs['r5'] = (p5, .6, 'obs', 0)

p1 = rf.vertex_to_poly(np.array([[2, 0], [3, 0], [3, -1], [2, -1]]))
regs['r6'] = (p1, 1, 'obs')
p2 = rf.vertex_to_poly(np.array([[2, -1], [3, -1], [3, -2], [2, -2]]))
regs['r7'] = (p2, 1, 'obs')
p3 = rf.vertex_to_poly(np.array([[2, -2], [3, -2], [3, -3], [2, -3]]))
regs['r8'] = (p3, 1, 'obs')
p4 = rf.vertex_to_poly(np.array([[2, -3], [3, -3], [3, -4], [2, -4]]))
regs['r9'] = (p4,1, 'obs', 0)
p5 = rf.vertex_to_poly(np.array([[2, -4], [3, -4], [3, -5], [2, -5]]))
regs['r10'] = (p5, .9, 'obs', 0)

a1 = rf.vertex_to_poly(np.array([[4, 0], [5, 0], [5, 1], [4, 1]]))
regs['a1'] = (a1, 0.9, 'sample1', 1)

a2 = rf.vertex_to_poly(np.array([[4, -3], [5, -3], [5, -2], [4, -2]]))
regs['a2'] = (a2, 0.1, 'sample2', 1)

output_color = {'r1': 'red', 'r2': 'red', 'r3': 'red', 'r4': 'red', 'r5': 'red', 'r6': 'red', 'r7': 'red',
                'r8': 'red', 'r9': 'red', 'r10': 'red',
                'a1': 'green','a2': 'blue', 'null': 'white'}
# Define Null regions with bounds of the space for null output with lowest priority
p = rf.vertex_to_poly(np.array([[-3, -5], [-3, 5], [5, -5], [5, 5]]))
regs['null'] = (p, 1.0, 'null')

# Construct belief-MDP for env
env = Env(regs)
print(env)


''' Configuration Parameters '''
random.seed(rand_seed)
np.random.seed(rand_seed)

print '''--- Setup Motion and Observation Models ---'''
# Define Motion and Observation Model
Wx = np.eye(2)
Wu = np.eye(2)
r2_bs = State_Space([-5, -5], [5, 5])
motion_model = Det_SI_Model(0.1)

print "---- Constructing ROADMAP ----"
fig = plt.figure(figsize=(14, 14), dpi=80, facecolor='w', edgecolor='k')
ax = fig.add_subplot(111, aspect='equal')
prm = SPaths(r2_bs, motion_model, Wx, Wu, regs, output_color, ax)
prm.make_nodes_edges(40, 3, init=np.array([[-4.5],[0]]))
prm.plot(ax)


print('-- Generate the DFA and the Product model----')
from best.fsa import Fsa
props = ['obs', 'sample1', 'sample2']
props = dict(zip(props, map(lambda x: 2 ** x, range(0, len(props)))))
print(props)
fsa = Fsa()
vars(Fsa)
fsa.g.add_node(0)
fsa.g.add_node('trap')
fsa.g.add_node(1)

fsa.g.add_edge(0,1, weight=0, input={props['sample1'],props['sample2'],props['sample2']+props['sample1']})
fsa.g.add_edge(0,0, weight=0, input={0})
fsa.g.add_edge(0,'trap', weight=0, input={props['obs']})

fsa.props=props
fsa.final = {1}
fsa.init = dict({0:1})


fsaform = Fsa()
form = '! obs U sample'
fsaform.from_formula(form)
vars(fsa.g)
print(fsaform.g.edges(data=True))
formula_fsa = dict()
formula_fsa['fsa'] = fsa
formula_fsa['init'] = dict({0:1})
formula_fsa['final'] = {1}
formula_fsa['prop'] = props

env.get_prop('r3')
prod_ = spec_Spaths(prm, formula_fsa,env,n=125)

import matplotlib.pyplot as plt





t_start = time.time()
print('--- Start Back-ups ---')

not_converged = True
i = 0
n = prod_.init[0]


opts = dict()
while not_converged:
    print('iteration', i)
    not_converged = prod_.full_back_up(opts)
    opt = np.unique(prod_.val[n].best_edge)
    if i>20:
        not_converged = False
    i += 1


print(time.time()-t_start)

from best.hVI_fsrm import plot_optimizers, simulate

nodes, edges, visited = plot_optimizers(prod_,ax)

prod_.prune(keep_list=visited)

simulate(prod_, regs)

plt.show()

print('--- Re-Start Back-ups ---')
t_start = time.time()

not_converged = True
i = 0
n = prod_.init[0]

opts = dict()
while not_converged:
    print('iteration', i)
    not_converged = prod_.full_back_up(opts)
    opt = np.unique(prod_.val[n].best_edge)
    if i>20:
        not_converged = False
    i += 1

print(time.time() - t_start)
nodes, edges, visited = plot_optimizers(prod_,ax)
