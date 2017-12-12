import core.data as data
import os
import random
from core.GameConfig import language

templatePath = os.path.join('data',language,'AttrTemplate.json')
templateData = data._loadjson(templatePath)

def getAttr(temName):
    temData = templateData[temName]
    age = getAge(temData)
    attrList = {
        age
    }
    return attrList

def getAge(temData):
    maxAge = int(temData['MaxAge'])
    miniAge = int(temData['MiniAge'])
    age = random.randint(miniAge,maxAge)
    return age
