# -*- coding:utf-8 _*-
"""
@Author  : Xiaoqi Cheng
@Time    : 2021/4/24 9:20
"""
from transformers import Rescale
from torch.autograd import Variable

import time, os, sys, warnings
import cv2
from dataloader import *
from model.model import LaneNet, compute_loss
from average_meter import *
warnings.filterwarnings('ignore')
# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEVICE = 'cuda'
save_folder = 'seg_result'
os.makedirs(save_folder, exist_ok=True)

# if __name__ == '__main__':
#     model = torch.load('checkpoints/200.pth', map_location=DEVICE)
#     model.eval()
#     sourceFileName='carvideo'
#     video_path = os.path.join("D:/1Learn/lanenet/lanenet/video/", sourceFileName+'.mp4')
#     times=0
#     frameFrequency=10
#     camera = cv2.VideoCapture(video_path)
#
#     while True:
#         res, imgori = camera.read()
#         times=times+1
#         if not res:
#             print('not res , not image')
#             break
#         if times%frameFrequency==0:
#             transform=transforms.Compose([Rescale((512, 256))])
#             imgori=transform(imgori)
#             toPIL = transforms.ToPILImage()
#             img = np.asarray(toPIL(imgori[:,:,[2,1,0]]))
#
#             img=np.transpose(img,(2,0,1))
#
#             img = np.expand_dims(img,0)
#
#             # print(img.shape)
#
#             imgdata=Variable(torch.from_numpy(img)).type(torch.FloatTensor).to(DEVICE)
#
#             # print(imgdata.size())
#             output=model(imgdata)
#             binary_seg_pred=output["binary_seg_pred"]
#
#             binary_seg_pred = binary_seg_pred.squeeze(0)
#             binary_seg_pred1=binary_seg_pred.to(torch.float32).cpu()
#
#             pic=toPIL(binary_seg_pred1)
#             imgx = cv2.cvtColor(np.asarray(pic),cv2.COLOR_RGB2BGR)
#             imgx[np.where((imgx!=[0, 0, 0]).all(axis=2))] = [255,255,255]
#
#             # print (imgori.shape)
#             # print (imgx.shape)
#
#             src7 = cv2.addWeighted(imgori,0,imgx,1,0)
#
#             final_img=cv2.resize(src7,(1280, 720))
#             final_img[np.where((final_img==[255, 255, 255]).all(axis=2))] = [0,0,255]
#
#             cv2.imshow('1',final_img)
#             print("frame"+str(times)+str(times))
#         if cv2.waitKey(20) & 0xff == 27:
#                 break
#     camera.release()
#     cv2.destroyAllWindows()


# if __name__ == '__main__':
#     dataset = 'data/training_data_example'
#     val_dataset_file = os.path.join(dataset, 'train.txt')
#     val_dataset = LaneDataSet(val_dataset_file, stage = 'val')
#     val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
#     model = torch.load('checkpoints/200.pth', map_location=DEVICE)
#     model.eval()
#     for batch_idx, (image_data, binary_label, instance_label) in enumerate(val_loader):
#         image_data, binary_label, instance_label = image_data.to(DEVICE),binary_label.type(torch.FloatTensor).to(DEVICE),instance_label.to(DEVICE)
#         with torch.set_grad_enabled(False):
#             # 预测，并可视化
#             # print(image_data.size())
#             net_output = model(image_data)
#             seg_logits = net_output["seg_logits"].cpu().numpy()[0]
#             # 背景为（0~50）黄色线为（51~200），白色线为（201~255）
#             result = (np.argmax(seg_logits, axis=0)*127).astype(np.uint8)       # 此处背景是0，黄色线是127，白色线是254
#             final = np.hstack((image_data.cpu().numpy()[0,0],result))
#             cv2.imshow('1',final)
#             # cv2.imshow('1', image_data.cpu().numpy()[0,0])
#             # cv2.imshow('1', result)
#             cv2.waitKey(100)
#             # cv2.imwrite(os.path.join(save_folder, '{0:04d}.png'.format(batch_idx)), result)
#             # fig, axs = plt.subplots(1,2)
#             # axs[0].imshow(image_data.cpu().numpy()[0,0])
#             # axs[1].imshow(result)
#             # plt.show()
#     cv2.destroyAllWindows()

if __name__ == '__main__':
    model = torch.load('checkpoints/200.pth', map_location=DEVICE)
    model.eval()

    imgori = cv2.imread('D:/1Learn\lanenet\lanenet\data/training_data_example\image/0000.png')
    transform=transforms.Compose([Rescale((512, 256))])
    imgori=transform(imgori)
    toPIL = transforms.ToPILImage()
    img = np.asarray(toPIL(imgori[:,:,[2,1,0]]))

    img=np.transpose(img,(2,0,1))

    img = np.expand_dims(img,0)

    # print(img.shape)

    imgdata=Variable(torch.from_numpy(img)).type(torch.FloatTensor).to(DEVICE)

    # print(imgdata.size())
    output=model(imgdata)
    binary_seg_pred=output["binary_seg_pred"]

    binary_seg_pred = binary_seg_pred.squeeze(0)
    binary_seg_pred1=binary_seg_pred.to(torch.float32).cpu()

    pic=toPIL(binary_seg_pred1)
    imgx = cv2.cvtColor(np.asarray(pic),cv2.COLOR_RGB2BGR)
    imgx[np.where((imgx!=[0, 0, 0]).all(axis=2))] = [255,255,255]

    # print (imgori.shape)
    # print (imgx.shape)

    src7 = cv2.addWeighted(imgori,0,imgx,1,0)

    final_img=cv2.resize(src7,(1280, 720))
    final_img[np.where((final_img==[255, 255, 255]).all(axis=2))] = [0,0,255]

    cv2.imshow('1',final_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

