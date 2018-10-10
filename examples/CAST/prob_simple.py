import polytope as pc
from collections import OrderedDict
import numpy as np

# problem definition
def get_prob():

  prob = {}
  prob['cas_T'] = 20

  prob['step_margin'] = 0.01
  prob['accept_margin'] = 0.8
  prob['reject_margin'] = 0.2

  prob['formula'] = '( F sampleA ) | ( F sampleB )'

  prob['xmin'] = [-1, -1]
  prob['xmax'] = [1, 1]
  prob['discretization'] = [4, 4]

  prob['cas_x0'] = np.array([-0.75, -0.75])
  prob['uav_x0'] = np.array([-0.75, -0.75])
  prob['uav_xT'] = np.array([-0.75, -0.75])

  regs = OrderedDict()
  regs['a1'] = (pc.box2poly(np.array([[0.5, 1], [-0.5, 0]])), 0.5, 'blue')
  regs['b1'] = (pc.box2poly(np.array([[-0.5, 0], [0.5, 1]])), 0.9, 'green')

  prob['regs'] = regs
  prob['env_x0'] = [1, 1]

  # what to reveal
  prob['REALMAP'] = [2, 0]    # SHOULD CONTAIN ZEROS AND TWOS
  return prob