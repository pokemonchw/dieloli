import os
import random
from script.Core import CacheContorl,GameConfig,GamePathConfig,TextLoading,JsonHandle

language = GameConfig.language
gamepath = GamePathConfig.gamepath
roleAttrPath = os.path.join(gamepath,'data',language,'RoleAttributes.json')
roleAttrData = JsonHandle._loadjson(roleAttrPath)

def getTemList():
    '''
    获取人物生成模板
    '''
    return TextLoading.getTextData(TextLoading.attrTemplatePath,'TemList')

def getFeaturesList():
    '''
    获取特征模板
    '''
    return roleAttrData['Features']

def getAgeTemList():
    '''
    获取年龄模板
    '''
    return list(TextLoading.getTextData(TextLoading.attrTemplatePath,'AgeTem').keys())

def getEngravingList():
    '''
    获取刻印列表
    '''
    return roleAttrData['Default']['Engraving']

def getClothing(sexId):
    '''
    按性别id获取服装模板
    Keyword arguments:
    sexId -- 指定性别id
    '''
    return TextLoading.getTextData(TextLoading.attrTemplatePath,'Equipment')[sexId]

def getSexItem(sexId):
    '''
    按性别id获取性道具模板
    Keyword arguments:
    sexId -- 指定性别id
    '''
    return TextLoading.getTextData(TextLoading.attrTemplatePath,'SexItem')[sexId]

def getGold():
    '''
    获取默认金钱数据
    '''
    return roleAttrData['Default']['Gold']

def getAttr(temName):
    '''
    按人物生成模板id生成人物属性
    Keyword arguments:
    temName -- 人物生成模板id
    '''
    TemList = TextLoading.getTextData(TextLoading.attrTemplatePath,'TemList')
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
    return {
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

def getAge(temName):
    '''
    按年龄模板id随机生成年龄数据
    Keyword arguments:
    temName -- 年龄模板id
    '''
    temData = TextLoading.getTextData(TextLoading.attrTemplatePath,'AgeTem')[temName]
    maxAge = int(temData['MaxAge'])
    miniAge = int(temData['MiniAge'])
    return random.randint(miniAge,maxAge)

def getHeight(temName,age,Features):
    '''
    按模板和年龄计算身高
    Keyword arguments:
    temName -- 人物生成模板
    age -- 人物年龄
    Features -- 人物特性数据
    '''
    temData = TextLoading.getTextData(TextLoading.attrTemplatePath,'HeightTem')[temName]
    initialHeight = random.uniform(temData[0],temData[1])
    age = int(age)
    expectHeightFix = 0
    figuresData = TextLoading.getTextData(TextLoading.rolePath,'Features')['Figure']
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

def getGrowthHeight(nowAge,expectHeight,developmentAge,expectAge):
    '''
    计算每日身高增长量
    Keyword arguments:
    nowAge -- 现在的年龄
    expectHeight -- 预期最终身高
    developmentAge -- 结束发育期时的年龄
    exceptAge -- 结束身高增长时的年龄
    '''
    if nowAge > developmentAge:
        nowHeight = expectHeight / 2
        judgeAge = expectAge - developmentAge
        growthHeight = nowHeight / (judgeAge * 365)
    else:
        judgeHeight = expectHeight / 2
        nowHeight = 0
        growthHeight = judgeHeight / (nowAge * 365)
    return {'GrowthHeight':growthHeight,'NowHeight':nowHeight}

def getBMI(temName):
    '''
    按体重比例模板生成BMI
    Keyword arguments:
    temName -- 体重比例模板id
    '''
    temData = TextLoading.getTextData(TextLoading.attrTemplatePath,'WeightTem')[temName]
    return random.uniform(temData[0],temData[1])

def getBodyFat(sex,temName):
    '''
    按性别和体脂率模板生成体脂率
    Keyword arguments:
    sex -- 性别
    temName -- 体脂率模板id
    '''
    if sex in ['Man','Asexual']:
        sexTem = 'Man'
    else:
        sexTem = 'Woman'
    temData = TextLoading.getTextData(TextLoading.attrTemplatePath,'BodyFatTem')[sexTem][temName]
    return random.uniform(temData[0],temData[1])

def getWeight(bmi,height):
    '''
    按bmi和身高计算体重
    Keyword arguments:
    bmi -- 身高体重比(BMI)
    height -- 身高
    '''
    height = height / 100
    return bmi * height * height

def getMeasurements(temName,height,weight,bodyFat,weightTem):
    '''
    计算角色三围
    Keyword arguments:
    temName -- 性别模板
    height -- 身高
    weight -- 体重
    bodyFat -- 体脂率
    weightTem -- 体重比例模板
    '''
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
    waistHipProportion = waist / hip
    waistHipProportionTem = TextLoading.getTextData(TextLoading.attrTemplatePath,'WaistHipProportionTem')[weightTem]
    waistHipProportionFix = random.uniform(0,waistHipProportionTem)
    waistHipProportion = waistHipProportion + waistHipProportionFix
    newHip = newWaist / waistHipProportion
    fix = newHip / hip
    bust = bust * fix
    return {"Bust": bust, "Waist": newWaist, 'Hip': newHip}

def getMaxHitPoint(temName):
    '''
    获取最大hp值
    Keyword arguments:
    temName -- hp模板
    '''
    temData = TextLoading.getTextData(TextLoading.attrTemplatePath,'HitPointTem')[temName]
    maxHitPoint = int(temData['HitPointMax'])
    addValue = random.randint(0,500)
    impairment = random.randint(0,500)
    return maxHitPoint + addValue - impairment

def getMaxManaPoint(temName):
    '''
    获取最大mp值
    Keyword arguments:
    temName -- mp模板
    '''
    temData = TextLoading.getTextData(TextLoading.attrTemplatePath,'ManaPointTem')[temName]
    maxManaPoint = int(temData['ManaPointMax'])
    addValue = random.randint(0,500)
    impairment = random.randint(0,500)
    return maxManaPoint + addValue - impairment

def getSexExperience(temName):
    '''
    按模板生成角色初始性经验
    Keyword arguments:
    temName -- 性经验模板
    '''
    temData = TextLoading.getTextData(TextLoading.attrTemplatePath,'SexExperience')[temName]
    mouthExperienceTemName = temData['MouthExperienceTem']
    bosomExperienceTemName = temData['BosomExperienceTem']
    vaginaExperienceTemName = temData['VaginaExperienceTem']
    clitorisExperienceTemName = temData['ClitorisExperienceTem']
    anusExperienceTemName = temData['AnusExperienceTem']
    penisExperienceTemName = temData['PenisExperienceTem']
    mouthExperienceList = TextLoading.getTextData(TextLoading.attrTemplatePath,'SexExperienceTem')['MouthExperienceTem'][mouthExperienceTemName]
    mouthExperience = random.randint(int(mouthExperienceList[0]),int(mouthExperienceList[1]))
    bosomExperienceList = TextLoading.getTextData(TextLoading.attrTemplatePath,'SexExperienceTem')['BosomExperienceTem'][bosomExperienceTemName]
    bosomExperience = random.randint(int(bosomExperienceList[0]),int(bosomExperienceList[1]))
    vaginaExperienceList = TextLoading.getTextData(TextLoading.attrTemplatePath,'SexExperienceTem')['VaginaExperienceTem'][vaginaExperienceTemName]
    vaginaExperience = random.randint(int(vaginaExperienceList[0]),int(vaginaExperienceList[1]))
    clitorisExperienceList = TextLoading.getTextData(TextLoading.attrTemplatePath,'SexExperienceTem')['ClitorisExperienceTem'][clitorisExperienceTemName]
    clitorisExperience = random.randint(int(clitorisExperienceList[0]),int(clitorisExperienceList[1]))
    anusExperienceList = TextLoading.getTextData(TextLoading.attrTemplatePath,'SexExperienceTem')['AnusExperienceTem'][anusExperienceTemName]
    anusExperience = random.randint(int(anusExperienceList[0]),int(anusExperienceList[1]))
    penisExperienceList = TextLoading.getTextData(TextLoading.attrTemplatePath,'SexExperienceTem')['PenisExperienceTem'][penisExperienceTemName]
    penisExperience = random.randint(int(penisExperienceList[0]),int(penisExperienceList[1]))
    return {
        'mouthExperience' : mouthExperience,
        'bosomExperience' : bosomExperience,
        'vaginaExperience' : vaginaExperience,
        'clitorisExperience' : clitorisExperience,
        'anusExperience' : anusExperience,
        'penisExperience':penisExperience,
    }

def getSexGrade(sexExperienceData):
    '''
    按性经验数据计算性经验等级
    Keyword arguments:
    sexExperienceData -- 性经验数据
    '''
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
    return {
        'mouthGrade' : mouthGrade,
        'bosomGrade' : bosomGrade,
        'vaginaGrade' : vaginaGrade,
        'clitorisGrade' : clitorisGrade,
        'anusGrade' : anusGrade,
        'penisGrade' : penisGrade
    }

# 计算等级
def judgeGrade(experience):
    '''
    按经验数值评定等级
    Keyword arguments:
    experience -- 经验数值
    '''
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

def setDefaultCache():
    '''
    生成默认特征数据并放入CacheContorl.featuresList
    '''
    featuresTemData = roleAttrData['defaultFeatures']
    cacheList = ['Age', "Chastity", 'Disposition', 'Courage', 'SelfConfidence', 'Friends', 'Figure',
                 'Sex', 'AnimalInternal', 'AnimalExternal', 'Charm'
                 ]
    for feature in cacheList:
        if feature in featuresTemData:
            CacheContorl.featuresList[feature] = featuresTemData[feature]

def setSexCache(sexName):
    '''
    生成性别对应特征数据并放入CacheContorl.featuresList
    Keyword arguments:
    sexName -- 性别id
    '''
    featuresTemData = roleAttrData['SexFeatures'][sexName]
    cacheList = ['Age', "Chastity", 'Disposition','Courage', 'SelfConfidence', 'Friends', 'Figure',
                 'Sex', 'AnimalInternal', 'AnimalExternal', 'Charm'
                 ]
    for feature in cacheList:
        if feature in featuresTemData:
            CacheContorl.featuresList[feature] = featuresTemData[feature]

def setAnimalCache(animalName):
    '''
    按动物类型追加特征数据
    Keyword arguments:
    animalName -- 动物名字
    '''
    animalData = roleAttrData["AnimalFeatures"][animalName]
    setAddFeatures(animalData)

def setAddFeatures(featuresData):
    '''
    追加特征存入CacheContorl.featuresList
    Keyword arguments:
    featuresData -- 追加的特征数据
    '''
    cacheList = ['Age', "Chastity", 'Disposition', "Courage", 'SelfConfidence', 'Friends', 'Figure',
                 'Sex', 'AnimalInternal', 'AnimalExternal', 'Charm'
                 ]
    for feature in cacheList:
        if feature in featuresData:
            CacheContorl.featuresList[feature] = featuresData[feature]

def setAttrDefault(characterId):
    '''
    为指定id角色生成默认属性
    Keyword arguments:
    characterId -- 角色id
    '''
    characterId = str(characterId)
    characterSex = CacheContorl.characterData['character'][characterId]['Sex']
    temData = getAttr(characterSex)
    for key in temData:
        CacheContorl.temporaryCharacter[key] = temData[key]
    CacheContorl.temporaryCharacter['Features'] = CacheContorl.featuresList.copy()

def initTemporaryCharacter():
    '''
    初始化角色模板
    '''
    CacheContorl.temporaryCharacter = CacheContorl.temporaryCharacterBak.copy()

def setAttrOver(characterId):
    '''
    将临时缓存内的数据覆盖入指定id的角色数据内
    Keyword arguments:
    characterId -- 角色id
    '''
    characterId = str(characterId)
    CacheContorl.characterData['character'][characterId] = CacheContorl.temporaryCharacter.copy()
    CacheContorl.featuresList = {}
