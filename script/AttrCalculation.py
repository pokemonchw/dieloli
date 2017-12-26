import core.data as data
import os
import random
import core.CacheContorl as cache
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

def getFeaturesStr(fList):
    featuresListStr = ''
    try:
        Age = fList['Age']
        if Age != '':
            featuresListStr = featuresListStr + '[' + Age + ']'
        else:
            pass
    except KeyError:
        pass
    try:
        Figure = fList['Figure']
        if Figure != '':
            featuresListStr = featuresListStr + '[' + Figure + ']'
        else:
            pass
    except KeyError:
        pass
    try:
        Sex = fList['Sex']
        if Sex != '':
            featuresListStr = featuresListStr + '[' + Sex + ']'
        else:
            pass
    except KeyError:
        pass
    try:
        AnimalInternal = fList['AnimalInternal']
        if AnimalInternal != '':
            featuresListStr = featuresListStr + '[' + AnimalInternal + ']'
        else:
            pass
    except KeyError:
        pass
    try:
        AnimalExternal = fList['AnimalExternal']
        if AnimalExternal != '':
            featuresListStr = featuresListStr + '[' + AnimalExternal + ']'
        else:
            pass
    except KeyError:
        pass
    try:
        Charm = fList['Charm']
        if Charm != '':
            featuresListStr = featuresListStr + '[' + Charm + ']'
        else:
            pass
    except KeyError:
        pass
    return featuresListStr

def setAnimalCache(animalName):
    animalData = roleAttrData["AnimalFeatures"][animalName]
    cacheSize = cache.temporaryObject['Features']
    try:
        Age = animalData['Age']
        cacheSize['Age'] = Age
    except KeyError:
        pass
    try:
        Figure = animalData['Figure']
        cacheSize['Figure'] = Figure
    except KeyError:
        pass
    try:
        Sex = animalData['Sex']
        cacheSize['Sex'] = Sex
    except KeyError:
        pass
    try:
        AnimalInternal = animalData['AnimalInternal']
        cacheSize['AnimalInternal'] = AnimalInternal
    except KeyError:
        pass
    try:
        AnimalExternal = animalData['AnimalExternal']
        cacheSize['AnimalExternal'] = AnimalExternal
    except KeyError:
        pass
    try:
        Charm = animalData['Charm']
        cacheSize['Charm'] = Charm
    except KeyError:
        pass
    pass