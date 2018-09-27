#!/usr/bin/env python

import socket
import struct
import struct
import numpy as np

import matlab.engine

import rospy
from geometry_msgs.msg import PoseStamped

from synthesize_plan import *
from policies import *

from problem_definition import get_prob

SIM = True

UDP_IP = '127.0.0.1'
UDP_PORT = 1560

MATLAB_QUADROTOR_PATH = r'/mnt/c/Users/petter/coding/quadrotor/lib'

MISSION_ALTITUDE = 0.5   # meters

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

def fit_poly_matlab(eng, t_ivals, xyz_ivals):
  t_ivals_m = matlab.double(list(t_ivals))
  x_ivals_m = matlab.double(list(xyz_ivals[0,:]))
  y_ivals_m = matlab.double(list(xyz_ivals[1,:]))
  z_ivals_m = matlab.double(list(xyz_ivals[2,:]))

  d_m = matlab.double([10])
  r_m = matlab.double([4])

  x_res_m = eng.optimize1d(t_ivals_m, x_ivals_m, d_m, r_m, nargout=2)
  y_res_m = eng.optimize1d(t_ivals_m, y_ivals_m, d_m, r_m, nargout=2)
  z_res_m = eng.optimize1d(t_ivals_m, z_ivals_m, d_m, r_m, nargout=2)

  udp_message_m = eng.pack_udp_message(t_ivals_m, x_res_m[0], y_res_m[0], z_res_m[0], nargout=1)
  udp_message = [udp_message_m[0][i] for i in range(udp_message_m.size[1])]
  udp_message_b = struct.pack('B', ord('F')) + struct.pack('{}B'.format(len(udp_message)), *udp_message)

  return udp_message_b

def send_target(eng, uav_current, uav_target, sock):

  xyz_ivals = np.array([[uav_current[0], uav_target[0]],
                        [uav_current[1], uav_target[1]],
                        [uav_current[2], uav_target[2]]])

  v_des = 0.3   # 0.3 m/s

  t_des = np.linalg.norm(xyz_ivals[:,0] - xyz_ivals[:,1]) / v_des
  t_ivals = np.array([0, t_des])

  print("changing target to", xyz_ivals[:,1])

  msg = fit_poly_matlab(eng, t_ivals, xyz_ivals)
  sock.sendto(msg, (UDP_IP, UDP_PORT))

def compute_ckhsum(msg):
  '''compute checksum for bytearray'''
  chksum = struct.pack('B', sum(msg) % 256)
  return chksum

class Planner(object):
  def __init__(self, matlab_eng):
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    np.random.seed(4)

    self.uav_policy, self.rover_policy = synthesize_plan(prob)
    self.s_map = prob['env_x0']            # state of map exploration (0 false, 1 unknown, 2 positive)

    self.cas_pos_pl = prob['cas_x0']
    self.uav_pos_pl = None                 # planning frame position
    self.uav_target_pl = prob['uav_x0']    # planning frame target

    self.matlab_eng = matlab_eng

    self.state = 0   # 0: not started,  1: executing,  2: landed                 

  def uav_callback(self, msg):
    uav_pos_ot = np.array([msg.pose.position.x, msg.pose.position.y])
    self.uav_pos_pl = optitrack_to_planning(uav_pos_ot)
    self.s_map = reveal_map(self.s_map, self.uav_pos_pl)

  def update_uav_target(self):
    # state machine
    # state = 0 : not started
    #         1 : executing
    #         2 : finished
    if self.state == 0:
      if self.uav_pos_pl is not None:
        self.state = 1
      
    if self.state == 1:

      if self.uav_policy.finished():
        # if finished, land
        cur_pos_ot = planning_to_optitrack(self.uav_pos_pl)
        uav_current = np.hstack([optitrack_to_uav(cur_pos_ot), MISSION_ALTITUDE])
        uav_target  = np.array([uav_current[0], uav_current[1], 0])
        send_target(self.matlab_eng, uav_current, uav_target, self.sock)

        self.state = 2

      else:
        # execute mission
        new_uav_target_pl, new_val = self.uav_policy(self.uav_pos_pl, self.s_map)

        if np.any(new_uav_target_pl != self.uav_target_pl):

          cur_pos_ot        = planning_to_optitrack(self.uav_pos_pl)
          new_uav_target_ot = planning_to_optitrack(new_uav_target_pl)

          print("estimated path remaining", new_val)
          print("current mission probability", self.rover_policy.get_value(self.cas_pos_pl, self.s_map))

          uav_current = np.hstack([optitrack_to_uav(cur_pos_ot), MISSION_ALTITUDE])
          uav_target  = np.hstack([optitrack_to_uav(new_uav_target_ot), MISSION_ALTITUDE])

          send_target(self.matlab_eng, uav_current, uav_target, self.sock)
          self.uav_target_pl = new_uav_target_pl

def main():
  plot_problem(prob)

  matlab_eng = matlab.engine.start_matlab()
  matlab_eng.addpath(MATLAB_QUADROTOR_PATH, nargout=0)

  planner = Planner(matlab_eng)
  rospy.Subscriber(UAV_POSE_TOPIC, PoseStamped, planner.uav_callback)
  rospy.init_node('best_planner', anonymous=True)
  rate = rospy.Rate(0.5)

  print("ready to control..")
  while not rospy.is_shutdown():
    planner.update_uav_target()
    rate.sleep()

  matlab_eng.quit()

if __name__ == '__main__':
  main()