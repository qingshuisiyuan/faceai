#coding=utf-8
#中文乱码处理

import cv2
import numpy
from PIL import Image, ImageDraw, ImageFont

# img = cv2.imread("img/xingye-1.png")

textSize = 25
fontFile = ImageFont.truetype("font/simsun.ttc", textSize, encoding = "utf-8")

def cv2ImgAddText(img, text, left, top, textColor=(13, 23, 64)):
    #判断是否OpenCV图片类型
    if (isinstance(img, numpy.ndarray)):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = fontFile
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(numpy.asarray(img), cv2.COLOR_RGB2BGR)