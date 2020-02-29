import os
import random
import time
from script.Core import CacheContorl,GameConfig,GamePathConfig,TextLoading,JsonHandle,ValueHandle
from script.Design import GameTime

language = GameConfig.language
gamepath = GamePathConfig.gamepath
roleAttrPath = os.path.join(gamepath,'data',language,'RoleAttributes.json')
roleAttrData = JsonHandle._loadjson(roleAttrPath)

def getTemList() -> dict:
    '''
    获取人物生成模板
    '''
    return TextLoading.getTextData(TextLoading.attrTemplatePath,'TemList')

def getFeaturesList() -> dict:
    '''
    获取特征模板
    '''
    return roleAttrData['Features']

def getAgeTemList() -> list:
    '''
    获取年龄模板
    '''
    return list(TextLoading.getTextData(TextLoading.attrTemplatePath,'AgeTem').keys())

def getEngravingList() -> dict:
    '''
    获取刻印列表
    '''
    return roleAttrData['Default']['Engraving']

def getSexItem(sexId:str) -> dict:
    '''
    按性别id获取性道具模板
    Keyword arguments:
    sexId -- 指定性别id
    '''
    return TextLoading.getTextData(TextLoading.attrTemplatePath,'SexItem')[sexId]

def getGold() -> int:
    '''
    获取默认金钱数据
    '''
    return roleAttrData['Default']['Gold']

def getAge(temName:str) -> int:
    '''
    按年龄模板id随机生成年龄数据
    Keyword arguments:
    temName -- 年龄模板id
    '''
    temData = TextLoading.getTextData(TextLoading.attrTemplatePath,'AgeTem')[temName]
    maxAge = int(temData['MaxAge'])
    miniAge = int(temData['MiniAge'])
    return random.randint(miniAge,maxAge)

def getHeight(temName:str,age:int,Features:dict) -> dict:
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
    if temName == 'Man' or 'Asexual':
        expectAge = random.randint(18,22)
        expectHeight = initialHeight / 0.2949
    else:
        expectAge = random.randint(13,17)
        expectHeight = initialHeight / 0.3109
    developmentAge = random.randint(4,6)
    growthHeightData = getGrowthHeight(age,expectHeight,developmentAge,expectAge)
    growthHeight = growthHeightData['GrowthHeight']
    nowHeight = growthHeightData['NowHeight']
    if age > expectAge:
        nowHeight = expectHeight
    else:
        nowHeight = 365 * growthHeight * age + nowHeight
    return {"NowHeight":nowHeight,"GrowthHeight":growthHeight,"ExpectAge":expectAge,"DevelopmentAge":developmentAge,"ExpectHeight":expectHeight}

def getChest(chestTem:str,birthday:dict):
    '''
    按罩杯模板生成人物最终罩杯，并按人物年龄计算当前罩杯
    Keyword arguments:
    chestTem -- 罩杯模板
    age -- 角色年龄
    '''
    targetChest = getRandNpcChest(chestTem)
    overAge = random.randint(14,18)
    overYear = birthday['year'] + overAge
    endDate = GameTime.getRandDayForYear(overYear)
    endDate = GameTime.systemTimeToGameTime(endDate)
    endDate = GameTime.gameTimeToDatetime(endDate)
    nowDate = CacheContorl.gameTime.copy()
    nowDate = GameTime.gameTimeToDatetime(nowDate)
    startDate = GameTime.gameTimeToDatetime(birthday)
    endDay = GameTime.countDayForDateToDate(startDate,endDate)
    nowDay = GameTime.countDayForDateToDate(startDate,nowDate)
    subChest = targetChest / endDay
    nowChest = subChest * nowDay
    if nowChest > subChest:
        nowChest = targetChest
    return {
        "TargetChest":targetChest,
        "NowChest":nowChest,
        "SubChest":subChest
    }

chestTemWeightData = TextLoading.getTextData(TextLoading.attrTemplatePath,'ChestWeightTem')
def getRandNpcChestTem() -> str:
    '''
    随机获取npc罩杯模板
    '''
    return ValueHandle.getRandomForWeight(chestTemWeightData)

def getRandNpcChest(chestTem:str) -> int:
    '''
    随机获取模板对应罩杯
    Keyword arguments:
    chestTem -- 罩杯模板
    '''
    chestScope = TextLoading.getTextData(TextLoading.attrTemplatePath,'ChestTem')[chestTem]
    return random.uniform(chestScope[0],chestScope[1])

def getRandNpcBirthDay(age:int):
    '''
    随机生成npc生日
    Keyword arguments:
    age -- 年龄
    '''
    nowYear = int(CacheContorl.gameTime['year'])
    nowMonth = int(CacheContorl.gameTime['month'])
    nowDay = int(CacheContorl.gameTime['day'])
    birthYear = nowYear - age
    date = time.localtime(GameTime.getRandDayForYear(birthYear))
    birthday = {
        "year":date[0],
        "month":date[1],
        "day":date[2]
    }
    if nowMonth < birthday['month'] or (nowMonth == birthday['month'] and nowDay < birthday['day']):
        birthday['year'] -= 1
    return birthday

def getGrowthHeight(nowAge:int,expectHeight:float,developmentAge:int,expectAge:int) -> dict:
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

def getBMI(temName:str) -> dict:
    '''
    按体重比例模板生成BMI
    Keyword arguments:
    temName -- 体重比例模板id
    '''
    temData = TextLoading.getTextData(TextLoading.attrTemplatePath,'WeightTem')[temName]
    return random.uniform(temData[0],temData[1])

def getBodyFat(sex:str,temName:str) -> float:
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

def getWeight(bmi:float,height:float) -> float:
    '''
    按bmi和身高计算体重
    Keyword arguments:
    bmi -- 身高体重比(BMI)
    height -- 身高
    '''
    height = height / 100
    return bmi * height * height

def getMeasurements(temName:str,height:float,weight:float,bodyFat:float,weightTem:str) -> dict:
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

def getMaxHitPoint(temName:str) -> int:
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

def getMaxManaPoint(temName:str) -> int:
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

def getSexExperience(temName:str) -> dict:
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

def getSexGrade(sexExperienceData:dict) -> dict:
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

def judgeGrade(experience:int) -> float:
    '''
    按经验数值评定等级
    Keyword arguments:
    experience -- 经验数值
    '''
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

def judgeAgeGroup(age:int):
    '''
    判断所属年龄段
    Keyword arguments:
    age -- 年龄
    '''
    ageGroup = TextLoading.getGameData(TextLoading.attrTemplatePath)['AgeTem']
    for ageTem in ageGroup:
        if int(age) >= int(ageGroup[ageTem]['MiniAge']) and int(age) < int(ageGroup[ageTem]['MaxAge']):
            return ageTem
    return 'YoundAdult'

def judgeChestGroup(chest:int):
    '''
    判断胸围差所属罩杯
    Keyword arguments:
    chest -- 胸围差
    '''
    chestGroup = TextLoading.getGameData(TextLoading.attrTemplatePath)['ChestTem']
    for chestTem in chestGroup:
        if int(chest) >= int(chestGroup[chestTem][0]) and int(chest) < chestGroup[chestTem][1]:
            return chestTem

def setAttrDefault(characterId:str):
    '''
    为指定id角色生成默认属性
    Keyword arguments:
    characterId -- 角色id
    '''
    characterSex = CacheContorl.characterData['character'][characterId].Sex
    temData = getAttr(characterSex)
    CacheContorl.temporaryCharacter.update(temData)
