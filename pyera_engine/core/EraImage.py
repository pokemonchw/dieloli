import os
import core.winframe as winframe
from tkinter import *
from core.pycfg import gamepath

textBox = winframe.textbox
imageData = {}

# 获取图片数据
def getImageData(imageName):
    imagePath = os.path.join(gamepath, 'image',imageName + '.png')
    imageData = PhotoImage(file=imagePath)
    return imageData

# 打印图片
def printImage(imageName):
    imageData[imageName] = getImageData(imageName)
    textBox.image_create(END,image = imageData[imageName])