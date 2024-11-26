#! /home/wyc/anaconda3/envs/pytorch2.0.0-cuda11.7/bin/python
# -*- coding: utf-8 -*-
"""
模拟机器人端接收控制指令
"""
import rospy
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose

def move(command):
    rospy.loginfo('机器人成功接收指令')

if __name__ == "__main__":

    rospy.init_node('robot_move_base')
     #订阅控制指令
    robot = rospy.Subscriber("cmd_vel",Twist,move,queue_size=10)
    rospy.spin()