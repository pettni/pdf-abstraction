import numpy as np

import cPickle as pickle

import rospy
from best_waypoints.srv import *

import sys
sys.path.append('../')

from best import midx_to_idx
from best.abstraction import Abstraction
from best.mdp import ParallelMDP
import rss18_functions as rf

INIT_STATE = [2, 2, 2, 2, 2]
TRUE_STATE = [0, 4, 4, 4, 4]

class CoptControl:

  def __init__(self, policy, prob):
    self.policy = policy
    self.prob = prob

    self.map_state = INIT_STATE[:]

  def update_map(self, x):
    '''reveal true map if state is appropriate'''
    conn_copt_env = rf.get_conn_copt_env(self.prob['regs'], self.prob['cop_sight'])

    map_meas = list(conn_copt_env(x))[0]

    for i in range(len(map_meas)):
      if self.map_state[i] in [0, 4]:
        continue
      if map_meas[i] == 1:  # weak meas
        if TRUE_STATE[i] == 4:
          self.map_state[i] = 3
        else:
          self.map_state[i] = 1
      if map_meas[i] == 2:
        self.map_state[i] = TRUE_STATE[i]

  def copter_callback(self, req):

    # read state
    x_curr = np.array([0., 0., 0.])
    x_curr[0] = req.current.x
    x_curr[1] = req.current.y
    x_curr[2] = req.current.z

    res_flag = req.reset_map

    if res_flag:
      # reset map state
      self.map_state = INIT_STATE[:]
    else:
      # update map state
      self.update_map(x_curr)

    # get waypoint
    s_env = midx_to_idx(self.map_state, [5] * len(self.prob['regs']))
    x_new, val = self.policy(x_curr, s_env)

    # return new waypoint
    ret = WaypointResponse()

    ret.target.x = x_new[0]
    ret.target.y = x_new[1]
    ret.target.z = x_new[2]

    ret.value = val

    ret.map_belief = self.map_state

    return ret


def copter_waypoints_server():
  rospy.init_node('copter_waypoints_server')

  with open("policies.pickle", "rb") as input_file:
      copter_policy, rover_policy, prob = pickle.load(input_file)

  cc = CoptControl(copter_policy, prob)

  s = rospy.Service('copter_waypoints', Waypoint, cc.copter_callback)

  print("copter waypoint server ready")

  rospy.spin()


if __name__ == '__main__':
  copter_waypoints_server()