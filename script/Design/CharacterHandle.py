import random
from script.Core import CacheContorl,ValueHandle,GameData,TextLoading,GamePathConfig,GameConfig
from script.Design import AttrCalculation,MapHandle,AttrText

language = GameConfig.language
gamepath = GamePathConfig.gamepath
featuresList = AttrCalculation.getFeaturesList()
sexList = list(TextLoading.getTextData(TextLoading.rolePath, 'Sex'))
ageTemList = list(TextLoading.getTextData(TextLoading.attrTemplatePath,'AgeTem'))
characterList = list(GameData._gamedata[language]['character'].keys())

def initCharacterList():
    '''
    初始生成所有npc数据
    '''
    initCharacterTem()
    i = 1
    for character in CacheContorl.npcTemData:
        initCharacter(i,character)
        i += 1
    initCharacterDormitory()
    initCharacterPosition()

def initCharacter(nowId,character):
    '''
    按id生成角色属性
    Keyword arguments:
    nowId -- 角色id
    character -- 角色生成模板数据
    '''
    AttrCalculation.initTemporaryCharacter()
    characterId = str(nowId)
    CacheContorl.characterData['character'][characterId] = CacheContorl.temporaryCharacter.copy()
    AttrCalculation.setDefaultCache()
    characterName = character['Name']
    characterSex = character['Sex']
    CacheContorl.characterData['character'][characterId]['Sex'] = characterSex
    defaultAttr = AttrCalculation.getAttr(characterSex)
    defaultAttr['Name'] = characterName
    defaultAttr['Sex'] = characterSex
    AttrCalculation.setSexCache(characterSex)
    defaultAttr['Features'] = CacheContorl.featuresList.copy()
    if 'MotherTongue' in character:
        defaultAttr['Language'][character['MotherTongue']] = 10000
        defaultAttr['MotherTongue'] = character['MotherTongue']
    else:
        defaultAttr['Language']['Chinese'] = 10000
    if 'Age' in character:
        ageTem = character['Age']
        characterAge = AttrCalculation.getAge(ageTem)
        defaultAttr['Age'] = characterAge
        characterAgeFeatureHandle(ageTem,characterSex)
        defaultAttr['Features'] = CacheContorl.featuresList.copy()
    elif 'Features' in character:
        AttrCalculation.setAddFeatures(character['Features'])
        defaultAttr['Features'] = CacheContorl.featuresList.copy()
    if 'Features' in character:
        height = AttrCalculation.getHeight(characterSex, defaultAttr['Age'],character['Features'])
    else:
        height = AttrCalculation.getHeight(characterSex, defaultAttr['Age'],{})
    defaultAttr['Height'] = height
    if 'Weight' in character:
        weightTemName = character['Weight']
    else:
        weightTemName = 'Ordinary'
    if 'BodyFat' in character:
        bodyFatTem = character['BodyFat']
    else:
        bodyFatTem = weightTemName
    bmi = AttrCalculation.getBMI(weightTemName)
    weight = AttrCalculation.getWeight(bmi, height['NowHeight'])
    defaultAttr['Weight'] = weight
    if defaultAttr['Age'] <= 18 and defaultAttr['Age'] >= 7:
        classGrade = str(defaultAttr['Age'] - 6)
        defaultAttr['Class'] = random.choice(CacheContorl.placeData["Classroom_" + classGrade])
    bodyFat = AttrCalculation.getBodyFat(characterSex,bodyFatTem)
    measurements = AttrCalculation.getMeasurements(characterSex, height['NowHeight'], weight,bodyFat,bodyFatTem)
    defaultAttr['Measirements'] = measurements
    defaultAttr['Knowledge'] = {}
    CacheContorl.temporaryCharacter.update(defaultAttr)
    CacheContorl.featuresList = {}
    CacheContorl.characterData['character'][characterId] = CacheContorl.temporaryCharacter.copy()
    CacheContorl.temporaryCharacter = CacheContorl.temporaryCharacterBak.copy()

def characterAgeFeatureHandle(ageTem,characterSex):
    '''
    按年龄模板生成角色特性数据
    Keyword arguments:
    ageTem -- 年龄模板
    characterSex -- 角色性别
    '''
    if ageTem == 'SchoolAgeChild':
        if characterSex == sexList[0]:
            CacheContorl.featuresList['Age'] = featuresList["Age"][0]
        elif characterSex == sexList[1]:
            CacheContorl.featuresList['Age'] = featuresList["Age"][1]
        else:
            CacheContorl.featuresList['Age'] = featuresList["Age"][2]
    elif ageTem == 'OldAdult':
        CacheContorl.featuresList['Age'] = featuresList["Age"][3]

def initCharacterTem():
    '''
    初始化角色模板数据
    '''
    npcData = getRandomNpcData()
    nowCharacterList = characterList.copy()
    npcData += [getDirCharacterTem(character) for character in nowCharacterList]
    CacheContorl.npcTemData = npcData

def getDirCharacterTem(character):
    '''
    获取预设角色模板数据
    '''
    return TextLoading.getCharacterData(character)['AttrTemplate']

randomNpcMax = int(GameConfig.random_npc_max)
randomTeacherProportion = int(GameConfig.proportion_teacher)
randomStudentProportion = int(GameConfig.proportion_student)
ageWeightData = {
    "Teacher":randomTeacherProportion,
    "Student":randomStudentProportion
}
ageWeightReginData = ValueHandle.getReginList(ageWeightData)
ageWeightReginList = list(map(int,ageWeightReginData.keys()))
def getRandomNpcData():
    '''
    生成所有随机npc的数据模板
    '''
    if CacheContorl.randomNpcList == []:
        ageWeightMax = 0
        for i in ageWeightData:
            ageWeightMax += int(ageWeightData[i])
        for i in range(0,randomNpcMax):
            nowAgeWeight = random.randint(-1,ageWeightMax - 1)
            nowAgeWeightRegin = ValueHandle.getNextValueForList(nowAgeWeight,ageWeightReginList)
            ageWeightTem = ageWeightReginData[str(nowAgeWeightRegin)]
            randomNpcSex = getRandNpcSex()
            randomNpcName = AttrText.getRandomNameForSex(randomNpcSex)
            randomNpcAgeTem = getRandNpcAgeTem(ageWeightTem)
            fatTem = getRandNpcFatTem(ageWeightTem)
            bodyFatTem = getRandNpcBodyFatTem(ageWeightTem,fatTem)
            randomNpcNewData = {
                "Name":randomNpcName,
                "Sex":randomNpcSex,
                "Age":randomNpcAgeTem,
                "Position":["0"],
                "AdvNpc":"1",
                "Weight":fatTem,
                "BodyFat":bodyFatTem
            }
            CacheContorl.randomNpcList.append(randomNpcNewData)
        return CacheContorl.randomNpcList

sexWeightData = TextLoading.getTextData(TextLoading.attrTemplatePath,'RandomNpcSexWeight')
sexWeightMax = 0
for i in sexWeightData:
    sexWeightMax += int(sexWeightData[i])
sexWeightReginData = ValueHandle.getReginList(sexWeightData)
sexWeightReginList = list(map(int,sexWeightReginData.keys()))
def getRandNpcSex():
    '''
    随机获取npc性别
    '''
    nowWeight = random.randint(0,sexWeightMax - 1)
    weightRegin = ValueHandle.getNextValueForList(nowWeight,sexWeightReginList)
    return sexWeightReginData[str(weightRegin)]

fatWeightData = TextLoading.getTextData(TextLoading.attrTemplatePath,'FatWeight')
def getRandNpcFatTem(agejudge):
    '''
    按人群年龄段体重分布比例随机生成体重模板
    Keyword arguments:
    agejudge -- 年龄段
    '''
    nowFatWeightData = fatWeightData[agejudge]
    nowFatTem = ValueHandle.getRandomForWeight(nowFatWeightData)
    return nowFatTem

bodyFatWeightData = TextLoading.getTextData(TextLoading.attrTemplatePath,'BodyFatWeight')
def getRandNpcBodyFatTem(ageJudge,bmiTem):
    '''
    按年龄段体脂率分布比例随机生成体脂率模板
    Keyword arguments:
    ageJudge -- 年龄段
    bmiTem -- bmi模板
    '''
    nowBodyFatData = bodyFatWeightData[ageJudge][bmiTem]
    return ValueHandle.getRandomForWeight(nowBodyFatData)

ageTemWeightData = TextLoading.getTextData(TextLoading.attrTemplatePath,'AgeWeight')
def getRandNpcAgeTem(agejudge):
    '''
    按年龄断随机生成npc年龄
    Keyword arguments:
    ageJudge -- 年龄段
    '''
    nowAgeWeightData  = ageTemWeightData[agejudge]
    nowAgeTem = ValueHandle.getRandomForWeight(nowAgeWeightData)
    return nowAgeTem

def getCharacterIndexMax():
    '''
    获取角色数量
    '''
    characterData = CacheContorl.characterData['character']
    characterDataMax = len(characterData.keys()) - 1
    return characterDataMax

def getCharacterIdList():
    '''
    获取角色id列表
    '''
    characterData = CacheContorl.characterData['character']
    return list(characterData.keys())

def initCharacterDormitory():
    '''
    分配角色宿舍
    '''
    characterData = {}
    for character in CacheContorl.characterData['character']:
        characterData[character] = CacheContorl.characterData['character'][character]['Age']
    characterData = [k[0] for k in sorted(characterData.items(),key=lambda x:x[1])]

def initCharacterPosition():
    '''
    初始化角色位置
    '''
    for character in CacheContorl.characterData['character']:
        characterPosition = CacheContorl.characterData['character'][character]['Position']
        MapHandle.characterMoveScene(characterPosition,'0',character)
