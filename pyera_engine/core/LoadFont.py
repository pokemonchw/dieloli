import PIL.ImageFont as imagefont
import os
from core.pycfg import gamepath

fontData = {}

def initFont():
    fontDirPath = os.path.join(gamepath,'font')
    fontPathList = []
    for fontFile in os.listdir(fontDirPath):
        fontPath = os.path.join(fontDirPath,fontFile)
        fontPathList.append(fontPath)
        index = str(fontFile).rfind('.')
        fontName = str(fontFile)[:index]
        varFont = imagefont.truetype(fontPath,18)
        fontData[fontName] = varFont
    pass

def loadFont(fontName):
    return fontData[fontName]