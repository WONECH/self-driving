#! /home/wyc/anaconda3/envs/pytorch2.0.0-cuda11.7/bin/python
# -*- coding: utf-8 -*-
"""
模拟机器人端发布图像消息
"""
import cv2
import rospy
import numpy as np
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image

def read_vd():

    vd = cv2.VideoCapture('/home/wyc/ws_self_driving_car/src/Yolov5_ros/yolov5_ros/yolov5_ros/media/carvideo.mp4')

    if vd.isOpened():
        open , frame = vd.read()
    else:
        open = False

    while open:
        ret , frame = vd.read()
        if frame is None:
            break
        if ret == True:
            img_pub.publish(CvBridge().cv2_to_imgmsg(frame,"bgr8"))
            cv2.imshow('robo_vd' , frame)
        if cv2.waitKey(20) & 0xff == 27:
            break
    vd.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":

    rospy.init_node('robot_cam', anonymous=True)
    img_pub = rospy.Publisher('robot_img', Image, queue_size=1)

    read_vd()
