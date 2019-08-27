import os
from tkinter import END
from PIL.ImageTk import PhotoImage
from script.Core import CacheContorl,GamePathConfig,MainFrame

gamepath = GamePathConfig.gamepath
textBox = MainFrame.textbox
imageData = {}
imageTextData = {}
imageLock = 0

def getImageData(imageName,imagePath=''):
    '''
    按路径读取图片数据并创建PhotoImage对象
    Keyword arguments:
    imageName -- 图片名字
    imagePath -- 图片路径 (default '')
    '''
    if imagePath == '':
        imagePath = os.path.join(gamepath, 'image', imageName + '.png')
    else:
        imagePath = os.path.join(gamepath,'image',imagePath,imageName + '.png')
    CacheContorl.imageid += 1
    return PhotoImage(file=imagePath)

def printImage(imageName,imagePath=''):
    '''
    绘制图片的内部实现，按图片id将图片加入绘制队列
    Keyword arguments:
    imageName -- 图片名字
    imagePath -- 图片路径 (default '')
    '''
    imageData[str(CacheContorl.imageid)] = getImageData(imageName, imagePath)
    textBox.image_create(END, image=imageData[str(CacheContorl.imageid)])
