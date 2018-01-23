import numpy as np

import cPickle as pickle

import rospy
from best_waypoints.srv import *

import sys
sys.path.append('../')

from best.abstraction import Abstraction
from best.mdp import ParallelMDP
import rss18_functions as rf

TRUE_STATE = [0, 4, 4, 4, 4]

class Environment:

	def __init__(self, regs, truestate):
		pass

class CoptControl:

	def __init__(self, policy):
		self.policy = policy


	def copter_callback(self, req):

		x_old = np.array([0., 0., 0.])
		x_old[0] = req.current.x
		x_old[1] = req.current.y
		x_old[2] = req.current.z

		s_env = 0  # get from env
		x_new, val = self.policy(x_old, s_env)

		ret = WaypointResponse()
		print ret
		ret.target.x = x_new[0]
		ret.target.y = x_new[1]
		ret.target.z = x_new[2]
		ret.value = val

		return ret


def copter_waypoints_server():
	rospy.init_node('copter_waypoints_server')

	with open("policies.pickle", "rb") as input_file:
	    copter_policy, rover_policy, prob = pickle.load(input_file)


	cc = CoptControl(copter_policy)
	env = Environment(prob['regs'], TRUE_STATE)

	s = rospy.Service('copter_waypoints', Waypoint, cc.copter_callback)

	print("copter waypoint server ready")

	rospy.spin()

if __name__ == '__main__':
	copter_waypoints_server()