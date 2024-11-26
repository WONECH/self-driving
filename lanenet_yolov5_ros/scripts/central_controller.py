#! /home/wyc/anaconda3/envs/pytorch2.0.0-cuda11.7/bin/python
# -*- coding: utf-8 -*-
"""
  中央调度系统
"""
import rospy
from yolov5_ros_msgs.msg import BoundingBox, BoundingBoxes
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose
from simple_pid import PID
import time

obj_Pose = Pose()
obj_Pose.position.x = 0
obj_Pose.position.y = 0
obj_Pose.position.z = 0
vel = Twist()
vel.linear.x = 0
vel.linear.y = 0
vel.linear.z = 0
vel.angular.x = 0
vel.angular.y = 0
vel.angular.z = 0

class Controller:
    def __init__(self):
        #订阅识别结果，位姿信息
        self.controller_sub = rospy.Subscriber("/yolov5/BoundingBoxes",BoundingBox,self.control,queue_size=10)
        #发布控制指令
        self.controller_pub = rospy.Publisher("cmd_vel",Twist, queue_size=5)
        #启动PID控制
        self.pid = PID()
        self.pid.sample_time = 0.01  # 每0.01s更新1次
        self.pid.setpoint = 540      # 目标值
        self.Kp=2.0
        self.Ki=0.1
        self.Kd=0.05

        self.s = 0 #识别框面积
        self.center_x = 0 #识别框中心x坐标
        self.center_y = 0 #识别框中心y坐标
        self.roadcenter_x = 540 #车道线中点x坐标

        #左右转标志的出现标志位
        self.left_flag = 0
        self.right_flag = 0

        #速度锁，1闭合，0开启
        self.vel_lock = 0

        #由于一次只处理一个标志，故即使存在对应的标志，在本次决策中可能判断为不存在该标识，因此需要一个计数标志，
        # 确保在足够多的接收信息中都没有该标志的出现，即可判断为该标志不存在
        self.count_p = 0 #行人
        self.count_rl = 0 #红灯
        self.count_rb = 0 #路障

    def control(self,result):#尚未完成决策算法
        rospy.loginfo("----------------------------")
        rospy.loginfo("标签:%s",result.Class)
        rospy.loginfo("左上角坐标：(%d,%d)",result.xmin,result.ymin)
        rospy.loginfo("右下角坐标：(%d,%d)",result.xmax,result.ymax)

        self.s = (result.xmax - result.xmin) * (result.ymax - result.ymin)#识别框面积
        self.center_x = int((result.xmax + result.xmin)/2)#识别框中心x坐标
        self.center_y = int((result.ymax + result.ymax)/2)#识别框中心y坐标

        #计数，计算接收次数，用于确保交通标志已经消失
        self.count_p = self.count_p + 1
        self.count_rl = self.count_rl + 1

        #交通标识决策
        #行人
        if((result.Class == 'pedestrian') & (self.s > 10000)):
            vel = Twist()
            vel.linear.x = 0
            self.vel_lock = 1 #速度锁关
            self.count_p = 0
        
        elif(self.count_p > 10 & self.count_rl > 10):
            vel = Twist()
            vel.linear.x = 1
            self.vel_lock = 1 #速度锁开

        #红灯
        if((result.Class == 'redlight') & (self.s > 10000)):
            vel = Twist()
            vel.linear.x = 0
            self.vel_lock = 1 #速度锁关
            self.count_rl = 0
        
        #绿灯
        if((result.Class == 'greenlight') & (self.s > 10000) & self.count_p > 10):
            vel = Twist()
            vel.linear.x = 1
            self.vel_lock = 0 #速度锁开

        #斑马线
        if((self.vel_lock == 0) & (result.Class == 'zebra_crossing') & (self.s > 100000)):
            vel = Twist()
            vel.linear.x = 0.5

        #路障
        if((self.vel_lock == 0) & (result.Class == 'roadblock') & (self.s > 10000)):
            self.pid.auto_mode = False #关闭PID控制器
            vel = Twist()
            vel.linear.x = 1
            if(self.road == 1): #如果在右车道则左传
                vel.angular.z = 1
                time.sleep(1)#延迟1s
                vel.angular.z = -1
                time.sleep(1)#延迟1s
            else:
                vel.angular.z = -1
                time.sleep(1)#延迟1s
                vel.angular.z = 1
                time.sleep(1)#延迟1s
            vel.angular.z = 0
            self.pid.auto_mode = True #打开PID控制器

        #左转
        if((result.Class == 'left') & (self.s > 100000) & (self.center_x > 100) & (self.center_x < 1000)):
            self.left_flag = 1
        #右转
        if((result.Class == 'right') & (self.s > 100000) & (self.center_x > 100) & (self.center_x < 1000)):
            self.right_flag = 1
        #左转或直行
        if((result.Class == 'left_or_straight') & (self.s > 100000) & (self.center_x > 100) & (self.center_x < 1000)):
            self.left_flag = 1
        #右转或直行
        if((result.Class == 'right_or_straight') & (self.s > 100000) & (self.center_x > 100) & (self.center_x < 1000)):
            self.right_flag = 1

        #车道线PID控制
        if((self.vel_lock == 0)):
            vel = Twist()
            vel.linear.z = self.pid(self.roadcenter_x)

        self.controller_pub.publish(vel)

if __name__ == "__main__":
    #初始化节点
    rospy.init_node("central_contraller")
    Controller()
    rospy.spin() #循环
