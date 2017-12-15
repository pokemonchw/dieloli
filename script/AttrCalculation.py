import core.data as data
import os
import random
from core.GameConfig import language
from core.pycfg import gamepath

templatePath = os.path.join(gamepath,'data',language,'AttrTemplate.json')
templateData = data._loadjson(templatePath)
roleAttrPath = os.path.join(gamepath,'data',language,'RoleAttributes.json')
roleAttrData = data._loadjson(roleAttrPath)

def getTemList():
    list = templateData['TemList']
    return list

def getFeaturesList():
    list = roleAttrData['Features']
    return list

def getAgeTemList():
    list = templateData["AgeTem"]["List"]
    return list

def getAttr(temName):
    temData = templateData[temName]
    ageTemName = temData["Age"]
    age = getAge(ageTemName)
    attrList = {
        'Age':age
    }
    return attrList

def getAge(temName):
    temData = templateData['AgeTem'][temName]
    maxAge = int(temData['MaxAge'])
    miniAge = int(temData['MiniAge'])
    age = random.randint(miniAge,maxAge)
    return age
