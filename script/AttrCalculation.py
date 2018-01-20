import core.data as data
import os
import random
import core.CacheContorl as cache
import script.TextLoading as textload
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
    hitPointTemName = temData["HitPoint"]
    maxHitPoint = getMaxHitPoint(hitPointTemName)
    manaPointTemName = temData["ManaPoint"]
    maxManaPoint = getMaxManaPoint(manaPointTemName)
    sexExperienceTemName = temData["SexExperience"]
    sexExperienceList = getSexExperience(sexExperienceTemName)
    attrList = {
        'Age':age,
        'MaxHitPoint':maxHitPoint,
        'HitPoint':maxHitPoint,
        'MaxManaPoint':maxManaPoint,
        'ManaPoint':maxManaPoint,
        'SexExperienceList':sexExperienceList
    }
    return attrList

def getAge(temName):
    temData = templateData['AgeTem'][temName]
    maxAge = int(temData['MaxAge'])
    miniAge = int(temData['MiniAge'])
    age = random.randint(miniAge,maxAge)
    return age

def getMaxHitPoint(temName):
    temData = templateData['HitPointTem'][temName]
    maxHitPoint = int(temData['HitPointMax'])
    addValue = random.randint(0,500)
    impairment = random.randint(0,500)
    maxHitPoint = maxHitPoint + addValue - impairment
    return maxHitPoint

def getMaxManaPoint(temName):
    temData = templateData['ManaPointTem'][temName]
    maxManaPoint = int(temData['ManaPointMax'])
    addValue = random.randint(0,500)
    impairment = random.randint(0,500)
    maxManaPoint = maxManaPoint + addValue - impairment
    return maxManaPoint

def getSexExperience(temName):
    temData = templateData['SexExperience'][temName]
    mouthExperienceTemName = temData['MouthExperienceTem']
    bosomExperienceTemName = temData['BosomExperienceTem']
    vaginaExperienceTemName = temData['VaginaExperienceTem']
    clitorisExperienceTemName = temData['ClitorisExperienceTem']
    anusExperienceTemName = temData['AnusExperienceTem']
    penisExperienceTemName = temData['PenisExperienceTem']
    mouthExperienceList = templateData['MouthExperienceTem'][mouthExperienceTemName]
    mouthExperience = random.randint(int(mouthExperienceList[0]),int(mouthExperienceList[1]))
    bosomExperienceList = templateData['BosomExperienceTem'][bosomExperienceTemName]
    bosomExperience = random.randint(int(bosomExperienceList[0]),int(bosomExperienceList[1]))
    vaginaExperienceList = templateData['VaginaExperienceTem'][vaginaExperienceTemName]
    vaginaExperience = random.randint(int(vaginaExperienceList[0]),int(vaginaExperienceList[1]))
    clitorisExperienceList = templateData['ClitorisExperienceTem'][clitorisExperienceTemName]
    clitorisExperience = random.randint(int(clitorisExperienceList[0]),int(clitorisExperienceList[1]))
    anusExperienceList = templateData['AnusExperienceTem'][anusExperienceTemName]
    anusExperience = random.randint(int(anusExperienceList[0]),int(anusExperienceList[1]))
    penisExperienceList = templateData['PenisExperienceTem'][penisExperienceTemName]
    penisExperience = random.randint(int(penisExperienceList[0]),int(penisExperienceList[1]))
    sexExperience = {
        'mouthExperience' : mouthExperience,
        'bosomExperience' : bosomExperience,
        'vaginaExperience' : vaginaExperience,
        'clitorisExperience' : clitorisExperience,
        'anusExperience' : anusExperience,
        'penisExperience':penisExperience,
    }
    return sexExperience

def getSexExperienceText(sexList,sexName):
    mouthExperience = sexList['mouthExperience']
    bosomExperience = sexList['bosomExperience']
    vaginaExperience = sexList['vaginaExperience']
    clitorisExperience = sexList['clitorisExperience']
    anusExperience = sexList['anusExperience']
    penisExperience = sexList['penisExperience']
    sexData = roleAttrData['Sex']
    sexExperienceText = []
    if sexName == sexData[0]:
        sexExperienceText = [
            textload.loadStageWordText('19') + mouthExperience,
            textload.loadStageWordText('20') + bosomExperience,
            textload.loadStageWordText('23') + anusExperience,
            textload.loadStageWordText('24') + penisExperience
        ]
    elif sexName == sexData[1]:
        sexExperienceText = [
            textload.loadStageWordText('19') + mouthExperience,
            textload.loadStageWordText('20') + bosomExperience,
            textload.loadStageWordText('21') + vaginaExperience,
            textload.loadStageWordText('22') + clitorisExperience,
            textload.loadStageWordText('23') + anusExperience
        ]
    elif sexName == sexData[2]:
        sexExperienceText = [
            textload.loadStageWordText('19') + mouthExperience,
            textload.loadStageWordText('20') + bosomExperience,
            textload.loadStageWordText('21') + vaginaExperience,
            textload.loadStageWordText('22') + clitorisExperience,
            textload.loadStageWordText('23') + anusExperience,
            textload.loadStageWordText('24') + penisExperience
        ]
    elif sexName == sexData[3]:
        sexExperienceText = [
            textload.loadStageWordText('19') + mouthExperience,
            textload.loadStageWordText('20') + bosomExperience,
            textload.loadStageWordText('23') + anusExperience
        ]
    return sexExperienceText

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