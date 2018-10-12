#!/usr/bin/env python

import struct
import numpy as np
import time

import rospy
from geometry_msgs.msg import PoseStamped

from planner import *
from policies import *
from rob_interface import RobCMD
from uav_interface import UAVCMD, is_landed

from prob_simple import get_prob

SIM = True

UDP_IP = '127.0.0.1'
UDP_PORT = 1560

MATLAB_QUADROTOR_PATH = r'/mnt/c/Users/petter/coding/quadrotor/lib'
UAV_ALTITUDE = 1.5  # m
UAV_SPEED = 0.5     # m/s

if SIM:
  UAV_POSE_TOPIC = '/MATLAB_UAV'
  ROB_POSE_TOPIC = '/MATLAB_ROB'
else:
  UAV_POSE_TOPIC = '/vrpn_client_node/AMBERUAV/pose'
  ROB_POSE_TOPIC = '/vrpn_client_node/AMBERSEG/pose'

prob = get_prob()

def reveal_map(s_map, uav_pos):
  ret = s_map
  for i, (name, item) in enumerate(prob['regs'].items()):
    if is_adjacent(item[0], uav_pos[0:2], 0):
      ret[i] = prob['REALMAP'][i]
  return ret

class Planner(object):

  def __init__(self, rob_cmd, uav_cmd):
    np.random.seed(4)

    self.rob_cmd = rob_cmd
    self.uav_cmd = uav_cmd

    self.uav_pos = None
    self.rob_pos = None

    self.uav_pol = None
    self.rob_pol = None

    self.s_map = prob['env_x0']         # state of map exploration (0 false, 1 unknown, 2 positive)

    self.change_state('plan_mission')   
    self.uavstate = 'landed'  
    self.mission_proba = 0.

  def uav_callback(self, msg):
    self.uav_pos = np.array([msg.pose.position.x,
                             msg.pose.position.y, 
                             msg.pose.position.z])
    self.s_map = reveal_map(self.s_map, self.uav_pos)                    # reveal hidden things..    

  def rob_callback(self, msg):

    siny_cosp = 2.0 * (msg.pose.orientation.w * msg.pose.orientation.z 
                       + msg.pose.orientation.x * msg.pose.orientation.y);
    cosy_cosp = 1.0 - 2.0 * (msg.pose.orientation.y * msg.pose.orientation.y
                             + msg.pose.orientation.z * msg.pose.orientation.z); 
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    self.rob_pos = np.array([msg.pose.position.x, msg.pose.position.y, yaw])  

  def change_state(self, newstate):
    
    # entry actions are defined here
    print('switching to', newstate)

    if newstate == 'plan_mission':
      self.rob_pol = None
      self.state = 'plan_mission'

    if newstate == 'execute_mission':
      self.state = 'execute_mission'

    if newstate == 'plan_exploration':
      self.uav_pol = None
      self.state = 'plan_exploration'

    if newstate == 'explore':
      self.rob_cmd.goto(self.rob_pos[0], self.rob_pos[1])  # stop here
      self.state = 'explore'

    if newstate == 'done':
      self.state = 'done'

  def step(self):
    
    # STATE MACHINE WITH FIVE STATES
    #
    # plan_mission, execute_mission, plan_exploration, explore, done

    if self.state == 'plan_mission':

      # during
      if self.rob_pos is not None:
        print("planning mission..")
        prob['cas_x0'] = np.array(self.rob_pos[0:2])
        self.rob_pol = plan_mission(prob)
      else:
        print("waiting for robot position data")

      # exit
      if self.rob_pol is not None:
        self.change_state('execute_mission')
    
    elif self.state == "execute_mission":
      # during
      aps = {}  # TODO: report these
      target, val = self.rob_pol(self.rob_pos[0:2], self.s_map, aps)
      print("current value", '{:.2f}'.format(val))
      if val > prob['accept_margin']:
        print("sending rob goto", target)
        self.rob_cmd.goto(target[0], target[1])

      # exit
      if self.rob_pol.finished():
        self.change_state('done')
      if not(val < prob['reject_margin'] or val > prob['accept_margin']):
        self.change_state('plan_exploration')
      if val == 0:
        self.change_state('done')

    elif self.state == 'plan_exploration':
      
      # during
      if self.rob_pos is not None:
        print("planning exploration..")
        prob['cas_x0'] = np.array(self.rob_pos[0:2])
        prob['uav_x0'] = np.array(self.rob_pos[0:2])
        prob['uav_xT'] = np.array(self.rob_pos[0:2])
        self.uav_pol = plan_exploration(prob, self.rob_pol) 
      else:
        print("waiting for position data")

      # exit
      if self.uav_pol is not None:
        self.change_state('explore')

    elif self.state == 'explore':

      # during
      if self.uavstate == 'flying' and self.uav_pol.finished():
        print("sending land")
        self.uav_cmd.land_on_platform(self.rob_pos, UAV_SPEED)
        
        if is_landed(self.uav_pos, self.rob_pos):
          time.sleep(3)
          self.uavstate = 'landed'
 
      elif self.uavstate == 'flying':
        target, val = self.uav_pol(self.uav_pos[0:2], self.s_map)
        print("sending uav target", target)
        self.uav_cmd.goto(target[0], target[1], UAV_ALTITUDE, UAV_SPEED)

      elif self.uavstate == 'landed':
        print("sending takeoff")
        self.uav_cmd.takeoff(UAV_SPEED)
        time.sleep(0.5)
        self.uavstate = 'flying'

      # exit
      if self.uavstate == 'landed' and self.uav_pol.finished():
        self.change_state('execute_mission')

    elif self.state == 'done':
      pass

    else:
      raise Exception("unknown state")


def main():
  plot_problem(prob)

  rob_cmd = RobCMD()
  uav_cmd = UAVCMD(UDP_IP, UDP_PORT)

  planner = Planner(rob_cmd, uav_cmd)

  rospy.Subscriber(UAV_POSE_TOPIC, PoseStamped, planner.uav_callback)
  rospy.Subscriber(ROB_POSE_TOPIC, PoseStamped, planner.rob_callback)
  
  rospy.init_node('best_planner', anonymous=True)
  
  rate = rospy.Rate(0.5)

  while not (rospy.is_shutdown() or planner.state == 'done'):
    planner.step()
    rate.sleep()

  rospy.signal_shutdown("planning ended") 
  return 0

if __name__ == '__main__':
  main()