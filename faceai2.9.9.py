#!/usr/bin/python
# -*- coding: UTF-8 -*-

#开发进度：预计3.0版本可以自由的按比例调整程序界面大小
#预计3.0版本后停止新功能开发进入软件维护阶段
#程序3.0版本后会进行树莓派硬件，显示屏效果测试阶段

"""
程序版本：2.9.9
可实现功能：识别人脸，识别人脸性别,识别人脸表情，给人脸添加头像挂件
本项目使用Python开发，兼容2.7及3.0以上版本
但在测试和实际使用中，python2.7对中文支持不好，会有中文乱码问题
建议使用python3运行本程序
Python兼容Linux，Windows，Mac等主流操作系统
在开发过程中使用了以下程序或环境：
Deepin Linux
Python
Pip
Numpy
OpenCV
keras
Dlib
face_recognition
tensorflow
Tesseract OCR
"""

# OpenCV版本的视频检测
import cv2

#性别识别，表情所需模块
from keras.models import load_model
import numpy as np
#中文乱码处理
import chineseText

#通过程序休眠降低CPU使用率从而在低配置机器上面使用
import time

#调用系统函数，清理终端显示内容
import os

# 定义绘制人脸边框的颜色RGB
color = (163, 214, 255)

#定义全局计数器
num = 0

#定义人脸表情中文代码
emotion_labels = {0: '生气',1: '厌恶',2: '恐惧',3: '开心',4: '难过',5: '惊喜',6: '平静'}

#定义人脸性别中文代码
gender_labels = {0: '女', 1: '男'}

#将所需文件加载解析到全局内存中
gender_classifier = load_model("classifier/gender_models/simple_CNN.81-0.96.hdf5")
emotion_classifier = load_model('classifier/emotion_models/simple_CNN.530-0.65.hdf5')
cap = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
imgCompose = cv2.imread("img/maozi-1.png") 

def discern(img):

    #理论上来讲，python3的int类型支持无限大，如果一直加下去，内存就会爆炸，所以还是到了一定数值的时候重启归零好点
    global num
    num+=1
    if(num == 1000):
        num = 0
        #清理系统终端内容
        os.system("clear")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # cap = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faceRects = cap.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50))
    if len(faceRects):
        # #框出正方形人脸
        #test
        # for faceRect in faceRects:
        #     x, y, w, h = faceRect
        #     cv2.rectangle(img, (x, y), (x + h, y + w), (0, 255, 0), 3)  

        #视频的每一帧都框出每一张人脸，包括眼睛和鼻子
        if (num % 2 == 0):
            for faceRect in faceRects:
                x, y, w, h = faceRect
                # 框出人脸
                cv2.rectangle(img, (x, y), (x + h, y + w), color, 2)
                # 左眼
                cv2.circle(img, (x + w // 4, y + h // 4 + 30), min(w // 8, h // 8),
                        color)
                #右眼
                cv2.circle(img, (x + 3 * w // 4, y + h // 4 + 30), min(w // 8, h // 8),
                        color)
                #嘴巴
                cv2.rectangle(img, (x + 3 * w // 8, y + 3 * h // 4),
                            (x + 5 * w // 8, y + 7 * h // 8), color)

        #识别男女性别
        if (num % 5 == 0):
            try:
                for (x, y, w, h) in faceRects:
                    face = img[(y - 60):(y + h + 60), (x - 30):(x + w + 30)]
                    face = cv2.resize(face, (48, 48))
                    face = np.expand_dims(face, 0)
                    face = face / 255.0
                    gender_label_arg = np.argmax(gender_classifier.predict(face))
                    gender = gender_labels[gender_label_arg]
                    cv2.rectangle(img, (x, y), (x + h, y + w), color, 2)
                    img = chineseText.cv2ImgAddText(img, gender, x + h, y, (224, 54, 54))
            except:
                print("错误代码：Oxf-001 已知错误！未检测到图像中的人脸或无法识别人脸性别")
        if (num % 3 == 0):
            #表情识别代码
            for (x, y, w, h) in faceRects:
                gray_face = gray[(y):(y + h), (x):(x + w)]
                gray_face = cv2.resize(gray_face, (48, 48))
                gray_face = gray_face / 255.0
                gray_face = np.expand_dims(gray_face, 0)
                gray_face = np.expand_dims(gray_face, -1)
                emotion_label_arg = np.argmax(emotion_classifier.predict(gray_face))
                emotion = emotion_labels[emotion_label_arg]
                img = chineseText.cv2ImgAddText(img, emotion, x + h * 0.3, y, (0, 0, 255))

        #头像挂件代码
        try:
            for faceRect in faceRects:
                # imgCompose = cv2.imread("img/maozi-1.png") 
                x, y, w, h = faceRect
                sp = imgCompose.shape
                imgComposeSizeH = int(sp[0]/sp[1]*w)
                if imgComposeSizeH>(y-20):
                    imgComposeSizeH=(y-20)
                imgComposeSize = cv2.resize(imgCompose,(w, imgComposeSizeH), interpolation=cv2.INTER_NEAREST)
                top = (y-imgComposeSizeH-20)
                if top<=0:
                    top=0
                rows, cols, channels = imgComposeSize.shape
                roi = img[top:top+rows,x:x+cols]

                img2gray = cv2.cvtColor(imgComposeSize, cv2.COLOR_RGB2GRAY)
                ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY) 
                mask_inv = cv2.bitwise_not(mask)

                img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

                img2_fg = cv2.bitwise_and(imgComposeSize, imgComposeSize, mask=mask)

                dst = cv2.add(img1_bg, img2_fg)
                img[top:top+rows, x:x+cols] = dst
        except:
            print("错误代码：Oxf-002 已知错误！sRGB配置文件！此错误不可消除，除非更改图片文件")

    cv2.imshow("Face recognition - q exit", img)

# 获取摄像头0表示第一个摄像头
video_capture = cv2.VideoCapture(0)
#有时候cap可能不成功的初始化摄像头设备，这种情况代码回报错，这时我们用cap.isOpened(),来检查是否成功初始化
while (True): 
    # 逐帧显示
    ret, img = video_capture.read()
    # cv2.imshow("Image", img)
    discern(img)
    time.sleep(0.05)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# 释放窗口资源
cv2.destroyAllWindows()