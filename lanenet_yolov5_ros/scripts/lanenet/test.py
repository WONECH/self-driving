#! /home/wyc/anaconda3/envs/pytorch2.0.0-cuda11.7/bin/python
# -*- coding:utf-8 _*-
"""
@Author  : Xiaoqi Cheng
@Time    : 2021/4/24 9:20
"""
from cv_bridge import CvBridge
from std_msgs.msg import String
from sensor_msgs.msg import Image as Image2
import rospy
from scripts.lanenet.transformers import Rescale
from torch.autograd import Variable
from postprocess import show_lane
import time, os, sys, warnings
import cv2
from dataloader import *
from model.model import LaneNet, compute_loss
from average_meter import *
warnings.filterwarnings('ignore')
# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEVICE = 'cpu'
save_folder = 'seg_result'
os.makedirs(save_folder, exist_ok=True)

if __name__ == '__main__':
    rospy.init_node('robot_cam')
    img_pub = rospy.Publisher('robot_img', Image2, queue_size=1)
    dataset = '/home/wyc/ws_self_driving_car/src/Yolov5_ros/yolov5_ros/yolov5_ros/data/training_data_example'
    val_dataset_file = os.path.join(dataset, 'train.txt')
    val_dataset = LaneDataSet(val_dataset_file, stage = 'val')
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    model = torch.load('checkpoints/200.pth', map_location='cpu')
    model.eval()
    imgs = os.listdir('/home/wyc/ws_self_driving_car/src/Yolov5_ros/yolov5_ros/yolov5_ros/data/training_data_example/image')
    imgs.sort(key=lambda x: int(x.split('.')[0]))
    count = 1
    for batch_idx, (image_data, binary_label, instance_label) in enumerate(val_loader):
        image_data, binary_label, instance_label = image_data.to(DEVICE),binary_label.type(torch.FloatTensor).to(DEVICE),instance_label.to(DEVICE)
        img = imgs[count]
        count=count+1
        with torch.set_grad_enabled(False):
            imgorg = Image.open('/home/wyc/ws_self_driving_car/src/Yolov5_ros/yolov5_ros/yolov5_ros/data/training_data_example/image' + '/' + img)
            imgorg = np.array(imgorg)
            r, g, b = cv2.split(imgorg)
            # 以b，g，r分量重新生成新图像
            imgorg = cv2.merge([b, g, r])
            # imgorg = cv2.resize(imgorg, (512, 256))
            # 预测，并可视化
            net_output = model(image_data)
            seg_logits = net_output["seg_logits"].cpu().numpy()[0]
            # 背景为（0~50）黄色线为（51~200），白色线为（201~255）
            result = (np.argmax(seg_logits, axis=0)*127).astype(np.uint8)       # 此处背景是0，黄色线是127，白色线是254
            result = cv2.resize(result, (1280, 720))
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            imgorg = show_lane(result , imgorg)
            img_pub.publish(CvBridge().cv2_to_imgmsg(imgorg,"bgr8"))
            # final = np.vstack((imgorg,result))
            # cv2.imshow('1',final)
            # if cv2.waitKey(1) & 0xff == 27:
            #     break
    cv2.destroyAllWindows()

