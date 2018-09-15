#!/usr/bin/env python

import socket
import struct
import numpy as np

import rospy
from geometry_msgs.msg import PoseStamped

from synthesize_plan import *
from policies import *

from problem_definition import get_prob

SIM = True

UDP_IP = '127.0.0.1'
UDP_PORT = 1560

if SIM:
  UAV_POSE_TOPIC = '/MATLAB_UAV'
else:
  UAV_POSE_TOPIC = '/vrpn_client_node/AMBERUAV/pose'

prob = get_prob()

def reveal_map(s_map, uav_pos):

  ret = s_map
  for i, (name, item) in enumerate(prob['regs'].items()):
    if is_adjacent(item[0], uav_pos, 0):
      ret[i] = prob['REALMAP'][i]
  return ret

# OPTITRACK: frame in which coordinates are received over ROS
# PLANNING : frame used in problem definition above
# UAV      : frame for copter motion planning

# Simulation: OPTITRACK = UAV, OPTITRACK != PLANNING

# Real      : OPTITRACK != UAV, OPTITRACK = PLANNING

def optitrack_to_planning(coordinate):
  if SIM:
    return coordinate + prob['uav_x0']
  else:
    return coordinate  

def planning_to_optitrack(coordinate):
  if SIM:
    return coordinate - prob['uav_x0']
  else:
    return coordinate

def optitrack_to_uav(coordinate):
  if SIM:
    return coordinate
  else:
    return coordinate - np.array([0, 0])

def uav_to_optitrack(coordinate):
  if SIM:
    return coordinate
  else:
    return coordinate + np.array([0, 0])

def send_target(xy_target, altitude, sock):
  target = np.array([xy_target[0], xy_target[1], altitude])

  print("changing target to", target)

  header = struct.pack('BB', ord('F'), ord('C'))

  uint32s = [2, 1, 1, 1] # num_t num_x num_y num_z
  uint32s_bytes = struct.pack('{}I'.format(len(uint32s)), *uint32s) 

  float32s = [0., 1., target[0], target[1], target[2]] # t0 t1 x y z
  float32s_bytes = struct.pack('{}f'.format(len(float32s)), *float32s)

  msg = header + uint32s_bytes + float32s_bytes
  msg += compute_ckhsum(msg[1:])

  sock.sendto(msg, (UDP_IP, UDP_PORT))

def compute_ckhsum(msg):
  '''compute checksum for bytearray'''
  chksum = struct.pack('B', sum(msg) % 256)
  return chksum

class Planner(object):
  def __init__(self):
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    self.uav_policy, self.rover_policy = synthesize_plan(prob)
    self.s_map = prob['env_x0']            # state of map exploration (0 false, 1 unknown, 2 positive)

    self.cas_pos_pl = prob['cas_x0']
    self.uav_pos_pl = None                 # planning frame position
    self.uav_target_pl = prob['uav_x0']    # planning frame target

  def uav_callback(self, msg):
    uav_pos_ot = np.array([msg.pose.position.x, msg.pose.position.y])
    self.uav_pos_pl = optitrack_to_planning(uav_pos_ot)
    self.s_map = reveal_map(self.s_map, self.uav_pos_pl)

  def update_uav_target(self):
    if self.uav_pos_pl is not None:
      new_uav_target_pl, new_val = self.uav_policy(self.uav_pos_pl, self.s_map)
      if np.any(new_uav_target_pl != self.uav_target_pl) or self.uav_policy.finished():
        new_uav_target_ot = planning_to_optitrack(new_uav_target_pl)
        print("current exploration probability", new_val)
        print("current mission probability", self.rover_policy.get_value(self.cas_pos_pl, self.s_map))
        send_target(optitrack_to_uav(new_uav_target_ot), 0.5 if not self.uav_policy.finished() else 0, self.sock)
        self.uav_target_pl = new_uav_target_pl

def main():

  plot_problem(prob)

  planner = Planner()
  rospy.Subscriber(UAV_POSE_TOPIC, PoseStamped, planner.uav_callback)
  rospy.init_node('best_planner', anonymous=True)
  rate = rospy.Rate(0.5)

  print("ready to control..")
  while not rospy.is_shutdown():
    planner.update_uav_target()
    rate.sleep()

if __name__ == '__main__':
  main()