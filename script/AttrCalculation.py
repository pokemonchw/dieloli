import core.data as data
import os
import random
from core.GameConfig import language
from core.pycfg import gamepath

templatePath = os.path.join(gamepath,'data',language,'AttrTemplate.json')
templateData = data._loadjson(templatePath)

def getTemList():
    list = templateData['TemList']
    return list

def getAttr(temName):
    temData = templateData[temName]
    age = getAge(temData)
    attrList = {
        'Age':age
    }
    return attrList

def getAge(temData):
    maxAge = int(temData['MaxAge'])
    miniAge = int(temData['MiniAge'])
    age = random.randint(miniAge,maxAge)
    return age
