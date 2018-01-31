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

# 获取模板列表
def getTemList():
    list = templateData['TemList']
    return list

# 获取特征数据
def getFeaturesList():
    list = roleAttrData['Features']
    return list

# 获取年龄模板
def getAgeTemList():
    list = templateData["AgeTem"]["List"]
    return list

# 获取刻应列表
def getEngravingList():
    list = roleAttrData['Default']['Engraving']
    return list

# 获取属性
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
    sexGradeList = getSexGrade(sexExperienceList)
    EngravingList = getEngravingList()
    attrList = {
        'Age':age,
        'MaxHitPoint':maxHitPoint,
        'HitPoint':maxHitPoint,
        'MaxManaPoint':maxManaPoint,
        'ManaPoint':maxManaPoint,
        'SexExperienceList':sexExperienceList,
        'SexGradeList':sexGradeList,
        'EngravingList':EngravingList
    }
    return attrList

# 获取年龄信息
def getAge(temName):
    temData = templateData['AgeTem'][temName]
    maxAge = int(temData['MaxAge'])
    miniAge = int(temData['MiniAge'])
    age = random.randint(miniAge,maxAge)
    return age

# 获取最大hp值
def getMaxHitPoint(temName):
    temData = templateData['HitPointTem'][temName]
    maxHitPoint = int(temData['HitPointMax'])
    addValue = random.randint(0,500)
    impairment = random.randint(0,500)
    maxHitPoint = maxHitPoint + addValue - impairment
    return maxHitPoint

# 获取最大mp值
def getMaxManaPoint(temName):
    temData = templateData['ManaPointTem'][temName]
    maxManaPoint = int(temData['ManaPointMax'])
    addValue = random.randint(0,500)
    impairment = random.randint(0,500)
    maxManaPoint = maxManaPoint + addValue - impairment
    return maxManaPoint

# 获取性经验数据
def getSexExperience(temName):
    temData = templateData['SexExperience'][temName]
    mouthExperienceTemName = temData['MouthExperienceTem']
    bosomExperienceTemName = temData['BosomExperienceTem']
    vaginaExperienceTemName = temData['VaginaExperienceTem']
    clitorisExperienceTemName = temData['ClitorisExperienceTem']
    anusExperienceTemName = temData['AnusExperienceTem']
    penisExperienceTemName = temData['PenisExperienceTem']
    mouthExperienceList = templateData['SexExperience']['MouthExperienceTem'][mouthExperienceTemName]
    mouthExperience = random.randint(int(mouthExperienceList[0]),int(mouthExperienceList[1]))
    bosomExperienceList = templateData['SexExperience']['BosomExperienceTem'][bosomExperienceTemName]
    bosomExperience = random.randint(int(bosomExperienceList[0]),int(bosomExperienceList[1]))
    vaginaExperienceList = templateData['SexExperience']['VaginaExperienceTem'][vaginaExperienceTemName]
    vaginaExperience = random.randint(int(vaginaExperienceList[0]),int(vaginaExperienceList[1]))
    clitorisExperienceList = templateData['SexExperience']['ClitorisExperienceTem'][clitorisExperienceTemName]
    clitorisExperience = random.randint(int(clitorisExperienceList[0]),int(clitorisExperienceList[1]))
    anusExperienceList = templateData['SexExperience']['AnusExperienceTem'][anusExperienceTemName]
    anusExperience = random.randint(int(anusExperienceList[0]),int(anusExperienceList[1]))
    penisExperienceList = templateData['SexExperience']['PenisExperienceTem'][penisExperienceTemName]
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

# 获取性器官敏感等级
def getSexGrade(sexExperienceData):
    mouthExperience = sexExperienceData['mouthExperience']
    bosomExperience = sexExperienceData['bosomExperience']
    vaginaExperience = sexExperienceData['vaginaExperience']
    clitorisExperience = sexExperienceData['clitorisExperience']
    anusExperience = sexExperienceData['anusExperience']
    penisExperience = sexExperienceData['penisExperience']
    mouthGrade = judgeGrade(mouthExperience)
    bosomGrade = judgeGrade(bosomExperience)
    vaginaGrade = judgeGrade(vaginaExperience)
    clitorisGrade = judgeGrade(clitorisExperience)
    anusGrade = judgeGrade(anusExperience)
    penisGrade = judgeGrade(penisExperience)
    sexGradeList = {
        'mouthGrade' : mouthGrade,
        'bosomGrade' : bosomGrade,
        'vaginaGrade' : vaginaGrade,
        'clitorisGrade' : clitorisGrade,
        'anusGrade' : anusGrade,
        'penisGrade' : penisGrade
    }
    return sexGradeList

# 计算等级
def judgeGrade(experience):
    experience = int(experience)
    grade = ''
    if experience < 50:
        grade = 'G'
    elif experience < 100:
        grade = 'F'
    elif experience < 200:
        grade = 'E'
    elif experience < 500:
        grade = 'D'
    elif experience < 1000:
        grade = 'C'
    elif experience < 2000:
        grade = 'B'
    elif experience < 5000:
        grade = 'A'
    elif experience < 10000:
        grade = 'S'
    elif experience >= 10000:
        grade = 'EX'
    return grade

# 设置默认特征
def setDefaultCache():
    featuresTemData = roleAttrData['defaultFeatures']
    cacheList = ['Age', "Chastity", 'Disposition', 'Courage', 'SelfConfidence', 'Friends', 'Figure',
                 'Sex', 'AnimalInternal', 'AnimalExternal', 'Charm'
                 ]
    for i in range(0,len(cacheList)):
        try:
            cacheText = featuresTemData[cacheList[i]]
            cache.featuresList[cacheList[i]] = cacheText
        except:
            pass

# 设置性别对应特征
def setSexCache(SexName):
    SexId = templateData['TemList'][SexName]
    featuresTemData = roleAttrData['SexFeatures'][SexId]
    cacheList = ['Age', "Chastity", 'Disposition','Courage', 'SelfConfidence', 'Friends', 'Figure',
                 'Sex', 'AnimalInternal', 'AnimalExternal', 'Charm'
                 ]
    for i in range(0,len(cacheList)):
        try:
            cacheText = featuresTemData[cacheList[i]]
            cache.featuresList[cacheList[i]] = cacheText
        except:
            pass

# 设置动物特征
def setAnimalCache(animalName):
    animalData = roleAttrData["AnimalFeatures"][animalName]
    cacheList = ['Age',"Chastity",'Disposition',"Courage",'SelfConfidence','Friends','Figure',
                        'Sex','AnimalInternal','AnimalExternal','Charm'
                        ]
    for i in range(0,len(cacheList)):
        try:
            cacheText = animalData[cacheList[i]]
            cache.featuresList[cacheList[i]] = cacheText
        except KeyError:
            pass
    pass