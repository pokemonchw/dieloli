import os
import core.winframe as winframe
from tkinter import *
from core.pycfg import gamepath
import PIL.Image as pilimage
import PIL.ImageDraw as pilimagedraw
import PIL.ImageTk as pilimagetk
import core.LoadFont as loadfont
import core.CacheContorl as cache

textBox = winframe.textbox
imageData = {}
imageTextData = {}

# 获取图片数据
def getImageData(imageName):
    imagePath = os.path.join(gamepath, 'image',imageName + '.png')
    imageData = PhotoImage(file=imagePath)
    return imageData

# 打印图片
def printImage(imageName):
    imageData[imageName] = getImageData(imageName)
    textBox.image_create(END,image = imageData[imageName])

def setImageText(text,fontName):
    font = loadfont.loadFont(fontName)
    textWidth, textHeight = font.getsize(text)
    canvas = pilimage.new('RGB',(textWidth, textHeight),'#000000')
    draw = pilimagedraw.Draw(canvas)
    draw.text((0,0),text,'#ffffff',font)
    imageTextData[str(cache.textid)] = pilimagetk.PhotoImage(canvas)
    pass

def printImageText(text,font):
    cache.textid = cache.textid + 1
    setImageText(text,font)
    textData = imageTextData[str(cache.textid)]
    textBox.image_create(END,image=textData)

def printImageCmd(text,fontName,cmdid):
    font = loadfont.loadFont(fontName)
    textWidth, textHeight = font.getsize(text)
    canvas = pilimage.new('RGB', (textWidth, textHeight), '#000000')
    draw = pilimagedraw.Draw(canvas)
    draw.text((0, 0), text, '#ffffff', font)
    cache.cmdData[cmdid] = pilimagetk.PhotoImage(canvas)
    textBox.image_create(END,image=cache.cmdData[cmdid])