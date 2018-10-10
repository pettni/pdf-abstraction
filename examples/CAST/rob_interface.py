import rospy
from cyberpod_ros.msg import command

class RobCMD:

  def __init__(self):
    self.pub = rospy.Publisher('/cyberpod_params', command, queue_size=10)

  def goto(self, x, y):
    msg = command([x, y])
    self.pub.publish(msg)