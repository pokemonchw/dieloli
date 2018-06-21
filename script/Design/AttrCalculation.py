import Core.GameData as data
import os
import random
from Core import CacheContorl,ValueHandle,GameConfig,GamePathConfig

language = GameConfig.language
gamepath = GamePathConfig.gamepath

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
    list = ValueHandle.dictKeysToList(templateData["AgeTem"])
    return list

# 获取刻应列表
def getEngravingList():
    list = roleAttrData['Default']['Engraving']
    return list

# 获取服装模板
def getClothing(sexId):
    clothingTem = templateData["Equipment"]
    clothingList = clothingTem[sexId]
    return clothingList

# 获取性道具模板
def getSexItem(sexid):
    SexItemTem = templateData["SexItem"]
    sexItemList = SexItemTem[sexid]
    return sexItemList

# 获取金钱模板
def getGold():
    gold = roleAttrData['Default']['Gold']
    return gold

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
    clothingList = getClothing(temName)
    sexItemList = getSexItem(temName)
    height = getHeight(temName,age)
    weightTemName = temData['Weight']
    weight = getWeight(weightTemName,height)
    measurements = getMeasurements(temName,height,weightTemName)
    gold = getGold()
    attrList = {
        'Age':age,
        'HitPointMax':maxHitPoint,
        'HitPoint':maxHitPoint,
        'ManaPointMax':maxManaPoint,
        'ManaPoint':maxManaPoint,
        'SexExperience':sexExperienceList,
        'SexGrade':sexGradeList,
        'Engraving':EngravingList,
        'Clothing':clothingList,
        'SexItem':sexItemList,
        'Height':height,
        'Weight':weight,
        'Measurements':measurements,
        'Gold':gold,
    }
    return attrList

# 获取年龄信息
def getAge(temName):
    temData = templateData['AgeTem'][temName]
    maxAge = int(temData['MaxAge'])
    miniAge = int(temData['MiniAge'])
    age = random.randint(miniAge,maxAge)
    return age

# 获取初始身高
def getHeight(temName,age):
    temData = templateData['HeightTem'][temName]
    initialHeight = random.uniform(temData[0],temData[1])
    age = int(age)
    if temName == 'Man' or 'Asexual':
        expectHeight = initialHeight / 0.2949
    else:
        expectHeight = initialHeight / 0.3109
    growthHeight = expectHeight / (22 * 365)
    if age > 22:
        nowHeight = expectHeight
    else:
        nowHeight = 365 * growthHeight * age
    return nowHeight

# 获取体重
def getWeight(temName,height):
    temData = templateData['Weight'][temName]
    bmi = random.uniform(temData[0],temData[1])
    height = height / 100
    weight = bmi * height * height
    return weight

# 获取三围
def getMeasurements(temName,height,weightTem):
    if temName == 'Man' or 'Asexual':
        bust = 51.76 / 100 * height
        waist = 42.79 / 100 * height
        hip = 52.07 / 100 * height
    else:
        bust = 52.35 / 100 * height
        waist = 41.34 / 100 * height
        hip = 57.78 / 100 * height
    measurements = {"Bust":bust,"Waist":waist,'Hip':hip}
    measurementsFix = [-4.9,4.9]
    if weightTem == 'Thin':
        measurementsFix = [-5,-14.9]
    elif weightTem == 'Overweight':
        measurementsFix = [5,14.9]
    elif weightTem == 'Fat':
        measurementsFix = [15,24.9]
    elif weightTem == 'SevereObesity':
        measurementsFix = [25,34.9]
    elif weightTem == 'ExtremelySevereObesity':
        measurementsFix = [35,44.9]
    for key in measurements:
        keyFix = random.uniform(measurementsFix[0],measurementsFix[1])
        measurements[key] = measurements[key] + keyFix
    return measurements

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
    mouthExperienceList = templateData['SexExperienceTem']['MouthExperienceTem'][mouthExperienceTemName]
    mouthExperience = random.randint(int(mouthExperienceList[0]),int(mouthExperienceList[1]))
    bosomExperienceList = templateData['SexExperienceTem']['BosomExperienceTem'][bosomExperienceTemName]
    bosomExperience = random.randint(int(bosomExperienceList[0]),int(bosomExperienceList[1]))
    vaginaExperienceList = templateData['SexExperienceTem']['VaginaExperienceTem'][vaginaExperienceTemName]
    vaginaExperience = random.randint(int(vaginaExperienceList[0]),int(vaginaExperienceList[1]))
    clitorisExperienceList = templateData['SexExperienceTem']['ClitorisExperienceTem'][clitorisExperienceTemName]
    clitorisExperience = random.randint(int(clitorisExperienceList[0]),int(clitorisExperienceList[1]))
    anusExperienceList = templateData['SexExperienceTem']['AnusExperienceTem'][anusExperienceTemName]
    anusExperience = random.randint(int(anusExperienceList[0]),int(anusExperienceList[1]))
    penisExperienceList = templateData['SexExperienceTem']['PenisExperienceTem'][penisExperienceTemName]
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
            CacheContorl.featuresList[cacheList[i]] = cacheText
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
            CacheContorl.featuresList[cacheList[i]] = cacheText
        except:
            pass

# 设置动物特征
def setAnimalCache(animalName):
    animalData = roleAttrData["AnimalFeatures"][animalName]
    setAddFeatures(animalData)
    pass

# 设置追加特征
def setAddFeatures(featuresData):
    cacheList = ['Age', "Chastity", 'Disposition', "Courage", 'SelfConfidence', 'Friends', 'Figure',
                 'Sex', 'AnimalInternal', 'AnimalExternal', 'Charm'
                 ]
    for i in range(0,len(cacheList)):
        try:
            cacheText = featuresData[cacheList[i]]
            CacheContorl.featuresList[cacheList[i]] = cacheText
        except KeyError:
            pass
    pass

# 创建角色默认属性
def setAttrDefault(playerId):
    temList = getTemList()
    playerId = str(playerId)
    playerSex = CacheContorl.playObject['object'][playerId]['Sex']
    temId = temList[playerSex]
    temData = getAttr(temId)
    for key in temData:
        CacheContorl.temporaryObject[key] = temData[key]
    CacheContorl.temporaryObject['Features'] = CacheContorl.featuresList.copy()

# 初始化角色属性模板
def initTemporaryObject():
    CacheContorl.temporaryObject = CacheContorl.temporaryObjectBak.copy()
    pass

# 确认角色最终属性生成
def setAttrOver(playerId):
    playerId = str(playerId)
    CacheContorl.playObject['object'][playerId] = CacheContorl.temporaryObject.copy()
    CacheContorl.featuresList = {}