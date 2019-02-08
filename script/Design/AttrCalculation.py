import os,random
from script.Core import CacheContorl,ValueHandle,GameConfig,GamePathConfig,TextLoading,GameData

language = GameConfig.language
gamepath = GamePathConfig.gamepath
roleAttrPath = os.path.join(gamepath,'data',language,'RoleAttributes.json')
roleAttrData = GameData._loadjson(roleAttrPath)

# 获取模板列表
def getTemList():
    list = TextLoading.getTextData(TextLoading.temId,'TemList')
    return list

# 获取特征数据
def getFeaturesList():
    list = roleAttrData['Features']
    return list

# 获取年龄模板
def getAgeTemList():
    list = ValueHandle.dictKeysToList(TextLoading.getTextData(TextLoading.temId,'AgeTem'))
    return list

# 获取刻应列表
def getEngravingList():
    list = roleAttrData['Default']['Engraving']
    return list

# 获取服装模板
def getClothing(sexId):
    clothingTem = TextLoading.getTextData(TextLoading.temId,'Equipment')
    clothingList = clothingTem[sexId]
    return clothingList

# 获取性道具模板
def getSexItem(sexid):
    sexItemTem = TextLoading.getTextData(TextLoading.temId,'SexItem')
    sexItemList = sexItemTem[sexid]
    return sexItemList

# 获取金钱模板
def getGold():
    gold = roleAttrData['Default']['Gold']
    return gold

# 获取属性
def getAttr(temName):
    TemList = TextLoading.getTextData(TextLoading.temId,'TemList')
    temData = TemList[temName]
    ageTemName = temData["Age"]
    age = getAge(ageTemName)
    sexExperienceTemName = temData["SexExperience"]
    sexExperienceList = getSexExperience(sexExperienceTemName)
    sexGradeList = getSexGrade(sexExperienceList)
    EngravingList = getEngravingList()
    clothingList = getClothing(temName)
    sexItemList = getSexItem(temName)
    height = getHeight(temName,age,{})
    weightTemName = temData['Weight']
    bmi = getBMI(weightTemName)
    weight = getWeight(bmi,height['NowHeight'])
    bodyFatTem = temData['BodyFat']
    bodyFat = getBodyFat(temName,bodyFatTem)
    measurements = getMeasurements(temName,height['NowHeight'],weight,bodyFat,bodyFatTem)
    hitPointTemName = temData["HitPoint"]
    maxHitPoint = getMaxHitPoint(hitPointTemName)
    manaPointTemName = temData["ManaPoint"]
    maxManaPoint = getMaxManaPoint(manaPointTemName)
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
        'BodyFat':bodyFat,
        'Measurements':measurements,
        'Gold':gold,
        'Language':{}
    }
    return attrList

# 获取年龄信息
def getAge(temName):
    temData = TextLoading.getTextData(TextLoading.temId,'AgeTem')[temName]
    maxAge = int(temData['MaxAge'])
    miniAge = int(temData['MiniAge'])
    age = random.randint(miniAge,maxAge)
    return age

# 获取初始身高
def getHeight(temName,age,Features):
    temData = TextLoading.getTextData(TextLoading.temId,'HeightTem')[temName]
    initialHeight = random.uniform(temData[0],temData[1])
    age = int(age)
    expectHeightFix = 0
    figuresData = TextLoading.getTextData(TextLoading.roleId,'Features')['Figure']
    try:
        if Features['Figure'] == figuresData[0]:
            expectHeightFix = 50
        elif Features['Figure'] == figuresData[1]:
            expectHeightFix = -50
    except KeyError:
        expectHeightFix = 0
    if temName == 'Man' or 'Asexual':
        expectAge = random.randint(18,22)
        expectHeight = initialHeight / 0.2949
    else:
        expectAge = random.randint(13,17)
        expectHeight = initialHeight / 0.3109
    expectHeight = expectHeight + expectHeightFix
    developmentAge = random.randint(4,6)
    growthHeightData = getGrowthHeight(age,expectHeight,developmentAge,expectAge)
    growthHeight = growthHeightData['GrowthHeight']
    nowHeight = growthHeightData['NowHeight']
    if age > expectAge:
        nowHeight = expectHeight
    else:
        nowHeight = 365 * growthHeight * age + nowHeight
    return {"NowHeight":nowHeight,"GrowthHeight":growthHeight,"ExpectAge":expectAge,"DevelopmentAge":developmentAge,"ExpectHeight":expectHeight}

# 获取每日身高增量
def getGrowthHeight(nowAge,expectHeight,developmentAge,expectAge):
    if nowAge > developmentAge:
        nowHeight = expectHeight / 2
        judgeAge = expectAge - developmentAge
        growthHeight = nowHeight / (judgeAge * 365)
    else:
        judgeHeight = expectHeight / 2
        nowHeight = 0
        growthHeight = judgeHeight / (nowAge * 365)
    return {'GrowthHeight':growthHeight,'NowHeight':nowHeight}

# 获取BMI
def getBMI(temName):
    temData = TextLoading.getTextData(TextLoading.temId,'WeightTem')[temName]
    bmi = random.uniform(temData[0],temData[1])
    return bmi

# 获取体脂率
def getBodyFat(sex,temName):
    if sex in ['Man','Asexual']:
        sexTem = 'Man'
    else:
        sexTem = 'Woman'
    temData = TextLoading.getTextData(TextLoading.temId,'BodyFatTem')[sexTem][temName]
    bodyFat = random.uniform(temData[0],temData[1])
    return bodyFat

# 获取体重
def getWeight(bmi,height):
    height = height / 100
    weight = bmi * height * height
    return weight

# 获取三围
def getMeasurements(temName,height,weight,bodyFat,weightTem):
    if temName == 'Man' or 'Asexual':
        bust = 51.76 / 100 * height
        waist = 42.79 / 100 * height
        hip = 52.07 / 100 * height
        newWaist = ((bodyFat / 100 * weight) + (weight * 0.082 + 34.89)) / 0.74
    else:
        bust = 52.35 / 100 * height
        waist = 41.34 / 100 * height
        hip = 57.78 / 100 * height
        newWaist = ((bodyFat / 100 * weight) + (weight * 0.082 + 44.74)) / 0.74
    bustWaistProportion = bust / waist
    bustHipProportion = bust / hip
    waistHipProportion = waist / hip
    waistHipProportionTem = TextLoading.getTextData(TextLoading.temId,'WaistHipProportionTem')[weightTem]
    waistHipProportionFix = random.uniform(0,waistHipProportionTem)
    waistHipProportion = waistHipProportion + waistHipProportionFix
    newHip = newWaist / waistHipProportion
    fix = newHip / hip
    bust = bust * fix
    measurements = {"Bust": bust, "Waist": newWaist, 'Hip': newHip}
    return measurements

# 获取最大hp值
def getMaxHitPoint(temName):
    temData = TextLoading.getTextData(TextLoading.temId,'HitPointTem')[temName]
    maxHitPoint = int(temData['HitPointMax'])
    addValue = random.randint(0,500)
    impairment = random.randint(0,500)
    maxHitPoint = maxHitPoint + addValue - impairment
    return maxHitPoint

# 获取最大mp值
def getMaxManaPoint(temName):
    temData = TextLoading.getTextData(TextLoading.temId,'ManaPointTem')[temName]
    maxManaPoint = int(temData['ManaPointMax'])
    addValue = random.randint(0,500)
    impairment = random.randint(0,500)
    maxManaPoint = maxManaPoint + addValue - impairment
    return maxManaPoint

# 获取性经验数据
def getSexExperience(temName):
    temData = TextLoading.getTextData(TextLoading.temId,'SexExperience')[temName]
    mouthExperienceTemName = temData['MouthExperienceTem']
    bosomExperienceTemName = temData['BosomExperienceTem']
    vaginaExperienceTemName = temData['VaginaExperienceTem']
    clitorisExperienceTemName = temData['ClitorisExperienceTem']
    anusExperienceTemName = temData['AnusExperienceTem']
    penisExperienceTemName = temData['PenisExperienceTem']
    mouthExperienceList = TextLoading.getTextData(TextLoading.temId,'SexExperienceTem')['MouthExperienceTem'][mouthExperienceTemName]
    mouthExperience = random.randint(int(mouthExperienceList[0]),int(mouthExperienceList[1]))
    bosomExperienceList = TextLoading.getTextData(TextLoading.temId,'SexExperienceTem')['BosomExperienceTem'][bosomExperienceTemName]
    bosomExperience = random.randint(int(bosomExperienceList[0]),int(bosomExperienceList[1]))
    vaginaExperienceList = TextLoading.getTextData(TextLoading.temId,'SexExperienceTem')['VaginaExperienceTem'][vaginaExperienceTemName]
    vaginaExperience = random.randint(int(vaginaExperienceList[0]),int(vaginaExperienceList[1]))
    clitorisExperienceList = TextLoading.getTextData(TextLoading.temId,'SexExperienceTem')['ClitorisExperienceTem'][clitorisExperienceTemName]
    clitorisExperience = random.randint(int(clitorisExperienceList[0]),int(clitorisExperienceList[1]))
    anusExperienceList = TextLoading.getTextData(TextLoading.temId,'SexExperienceTem')['AnusExperienceTem'][anusExperienceTemName]
    anusExperience = random.randint(int(anusExperienceList[0]),int(anusExperienceList[1]))
    penisExperienceList = TextLoading.getTextData(TextLoading.temId,'SexExperienceTem')['PenisExperienceTem'][penisExperienceTemName]
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
def setSexCache(sexName):
    featuresTemData = roleAttrData['SexFeatures'][sexName]
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
    playerId = str(playerId)
    playerSex = CacheContorl.playObject['object'][playerId]['Sex']
    temData = getAttr(playerSex)
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
