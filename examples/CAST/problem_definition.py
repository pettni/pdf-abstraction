import polytope as pc
from collections import OrderedDict
import numpy as np

# problem definition
def get_prob():

	prob = {}
	prob['cas_T'] = 20
	prob['uav_T'] = 30
	prob['cas_x0'] = np.array([0.25, 0.75])
	prob['uav_x0'] = np.array([0.25, 3.25])
	prob['uav_xT'] = np.array([0.25, 3.26])

	prob['prob_margin'] = 0.01

	prob['formula'] = '( ( ! fail U sampleA ) & ( ! fail U sampleB ) ) | ( ! fail U ( sampleC ) )'

	prob['xmin'] = [0, 0]
	prob['xmax'] = [6, 4]
	prob['discretization'] = [12, 8]

	regs = OrderedDict()
	regs['r1'] = (pc.box2poly(np.array([[3, 6], [1, 1.5]])), 0.5, 'red')
	regs['r2'] = (pc.box2poly(np.array([[3, 6], [2.5, 3]])), 1, 'red')
	regs['r3'] = (pc.box2poly(np.array([[4, 4.5], [3, 4]])), 0.5, 'red')
	regs['a1'] = (pc.box2poly(np.array([[5.5, 6], [1.5, 2.5]])), 0.5, 'green')
	regs['b1'] = (pc.box2poly(np.array([[5.5, 6], [0, 0.5]])), 0.5, 'blue')
	regs['c1'] = (pc.box2poly(np.array([[5.5, 6], [3.5, 4]])), 0.5, 'blue')

	prob['regs'] = regs
	prob['env_x0'] = [0 if  reg[1] in [0,1] else 1 for reg in regs.values()]

	# what to reveal
	prob['REALMAP'] = [0, 0, 2, 0, 0, 2]
	return prob