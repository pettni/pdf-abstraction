import numpy as np
import rospy
from best_waypoints.srv import *

import matplotlib.pyplot as plt

rospy.wait_for_service('copter_waypoints')

handle = rospy.ServiceProxy('copter_waypoints', Waypoint)

x_copter = np.array([-0.5, -4.5, 2]).reshape((1,3))

for i in range(200):

  x_curr = x_copter[-1, :].flatten()

  req = WaypointRequest()
  req.current.x = x_curr[0]
  req.current.y = x_curr[1]
  req.current.z = x_curr[2]

  req.reset_map = False

  res = handle(req)
  
  print res

  x_next = np.zeros(3)
  x_next[0] = res.target.x
  x_next[1] = res.target.y
  x_next[2] = res.target.z

  x_del = 0.1*(x_next - x_curr)/np.linalg.norm(x_next - x_curr)


  x_copter = np.vstack([x_copter, x_curr + x_del])

fig = plt.figure()
ax = plt.Axes(fig, [0.,0.,1.,1.])
fig.add_axes(ax)

ax.plot(x_copter[:, 0], x_copter[:, 1], color='blue', linewidth=2)
