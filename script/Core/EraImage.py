import os
from tkinter import PhotoImage,END
from script.Core import CacheContorl,GamePathConfig,MainFrame

gamepath = GamePathConfig.gamepath
textBox = MainFrame.textbox
imageData = {}
imageTextData = {}
imageLock = 0

# 获取图片数据
def getImageData(imageName,imagePath=''):
    if imagePath == '':
        imagePath = os.path.join(gamepath, 'image', imageName + '.png')
    else:
        imagePath = os.path.join(gamepath,'image',imagePath,imageName + '.png')
    image = PhotoImage(file=imagePath)
    CacheContorl.imageid = CacheContorl.imageid + 1
    return image

# 打印图片
def printImage(imageName,imagePath=''):
    imageData[str(CacheContorl.imageid)] = getImageData(imageName, imagePath)
    textBox.image_create(END, image=imageData[str(CacheContorl.imageid)])