import os
from tkinter import PhotoImage,END
from script.Core import CacheContorl,GamePathConfig,MainFrame

gamepath = GamePathConfig.gamepath
textBox = MainFrame.textbox
imageData = {}
imageTextData = {}
imageLock = 0

'''
内部接口:读取图片数据
@imageName 图片名字
@imaagePath 图片路径
'''
def getImageData(imageName,imagePath=''):
    if imagePath == '':
        imagePath = os.path.join(gamepath, 'image', imageName + '.png')
    else:
        imagePath = os.path.join(gamepath,'image',imagePath,imageName + '.png')
    CacheContorl.imageid += 1
    return PhotoImage(file=imagePath)

'''
将图片插入输出队列
@imageName 图片名字
@imagePath 图片路径
'''
def printImage(imageName,imagePath=''):
    imageData[str(CacheContorl.imageid)] = getImageData(imageName, imagePath)
    textBox.image_create(END, image=imageData[str(CacheContorl.imageid)])
