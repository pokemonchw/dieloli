import os,random,multiprocessing,datetime
from script.Core import CacheContorl,ValueHandle,GameData,TextLoading,GamePathConfig,GameConfig
from script.Design import AttrCalculation,MapHandle,AttrText

language = GameConfig.language
gamepath = GamePathConfig.gamepath
featuresList = AttrCalculation.getFeaturesList()
sexList = list(TextLoading.getTextData(TextLoading.roleId, 'Sex'))
ageTemList = list(TextLoading.getTextData(TextLoading.temId,'AgeTem'))

characterInitList = []
# 初始化角色数据
def initCharacterList():
    initCharacterTem()
    characterList = CacheContorl.npcTemData
    processesMax = GameConfig.process_pool
    i = 0
    characterPool = multiprocessing.Pool(processes=processesMax)
    time1 = datetime.datetime.now()
    for character in characterList:
        characterPool.apply_async(initCharacter, args=(i,))
        i += 1
        characterInitList.append(str(i))
    time2 = datetime.datetime.now()
    print(time2 - time1)
    characterPool.close()
    characterPool.join()
    while characterInitList == []:
        initPlayerPosition()

# 按id生成角色属性
def initCharacter(nowId,character):
    AttrCalculation.initTemporaryObject()
    playerId = str(nowId)
    CacheContorl.playObject['object'][playerId] = CacheContorl.temporaryObject.copy()
    AttrCalculation.setDefaultCache()
    characterName = character['Name']
    characterSex = character['Sex']
    CacheContorl.playObject['object'][playerId]['Sex'] = characterSex
    defaultAttr = AttrCalculation.getAttr(characterSex)
    defaultAttr['Name'] = characterName
    defaultAttr['Sex'] = characterSex
    AttrCalculation.setSexCache(characterSex)
    defaultAttr['Features'] = CacheContorl.featuresList.copy()
    motherTongue = {
        "Level":5,
        "Exp":0
    }
    if 'MotherTongue' in character:
        defaultAttr['Language'][character['MotherTongue']] = motherTongue
        defaultAttr['MotherTongue'] = character['MotherTongue']
    else:
        defaultAttr['Language']['Chinese'] = motherTongue
    if 'Age' in character:
        ageTem = character['Age']
        characterAge = AttrCalculation.getAge(ageTem)
        defaultAttr['Age'] = characterAge
        characterAgeFeatureHandle(ageTem,characterSex)
        defaultAttr['Features'] = CacheContorl.featuresList.copy()
    elif 'Features' in character:
        AttrCalculation.setAddFeatures(character['Features'])
        defaultAttr['Features'] = CacheContorl.featuresList.copy()
    temList = AttrCalculation.getTemList()
    if 'Features' in character:
        height = AttrCalculation.getHeight(characterSex, defaultAttr['Age'],character['Features'])
    else:
        height = AttrCalculation.getHeight(characterSex, defaultAttr['Age'],{})
    defaultAttr['Height'] = height
    if 'Weight' in character:
        weightTemName = character['Weight']
    else:
        weightTemName = 'Ordinary'
    bmi = AttrCalculation.getBMI(weightTemName)
    weight = AttrCalculation.getWeight(bmi, height['NowHeight'])
    defaultAttr['Weight'] = weight
    schoolClassDataPath = os.path.join(gamepath,'data',language,'SchoolClass.json')
    schoolClassData = GameData._loadjson(schoolClassDataPath)
    if defaultAttr['Age'] <= 18 and defaultAttr['Age'] >= 7:
        classGradeMax = len(schoolClassData['Class'].keys())
        classGrade = str(defaultAttr['Age'] - 6)
        if int(classGrade) > classGradeMax:
            classGrade = str(classGradeMax)
        defaultAttr['Class'] = random.choice(schoolClassData['Class'][classGrade])
    else:
        defaultAttr['Office'] = str(random.randint(0,12))
    measurements = AttrCalculation.getMeasurements(characterSex, height['NowHeight'], weightTemName)
    defaultAttr['Measurements'] = measurements
    for keys in defaultAttr:
        CacheContorl.temporaryObject[keys] = defaultAttr[keys]
    CacheContorl.featuresList = {}
    CacheContorl.playObject['object'][playerId] = CacheContorl.temporaryObject.copy()
    CacheContorl.temporaryObject = CacheContorl.temporaryObjectBak.copy()
    characterInitList.remove(str(nowId))

# 处理角色年龄特性
def characterAgeFeatureHandle(ageTem,characterSex):
    characterAge = AttrCalculation.getAge(ageTem)
    if ageTem == 'SchoolAgeChild':
        if characterSex == sexList[0]:
            CacheContorl.featuresList['Age'] = featuresList["Age"][0]
        elif characterSex == sexList[1]:
            CacheContorl.featuresList['Age'] = featuresList["Age"][1]
        else:
            CacheContorl.featuresList['Age'] = featuresList["Age"][2]
    elif ageTem == 'OldAdult':
        CacheContorl.featuresList['Age'] = featuresList["Age"][3]

# 初始化角色数据
def initCharacterTem():
    characterListPath = os.path.join(gamepath,'data',language,'character')
    characterList = GameData.getPathList(characterListPath)
    npcData = getRandomNpcData()
    for i in characterList:
        characterAttrTemPath = os.path.join(characterListPath,i,'AttrTemplate.json')
        characterData = GameData._loadjson(characterAttrTemPath)
        npcData.append(characterData)
    CacheContorl.npcTemData = npcData

randomNpcMax = int(GameConfig.random_npc_max)
randomTeacherProportion = int(GameConfig.proportion_teacher)
randomStudentProportion = int(GameConfig.proportion_student)
ageWeightData = {
    "Teacher":randomTeacherProportion,
    "Student":randomStudentProportion
}
ageWeightReginData = ValueHandle.getReginList(ageWeightData)
ageWeightReginList = ValueHandle.getListKeysIntList(list(ageWeightReginData.keys()))
# 获取随机npc数据
def getRandomNpcData():
    if CacheContorl.randomNpcList == []:
        ageWeightMax = 0
        for i in ageWeightData:
            ageWeightMax += int(ageWeightData[i])
        for i in range(0,randomNpcMax):
            nowAgeWeight = random.randint(0,ageWeightMax - 1)
            nowAgeWeightRegin = next(x for x in ageWeightReginList if x > nowAgeWeight)
            ageWeightTem = ageWeightReginData[str(nowAgeWeightRegin)]
            randomNpcSex = getRandNpcSex()
            randomNpcName = AttrText.getRandomNameForSex(randomNpcSex)
            randomNpcAgeTem = getRandNpcAgeTem(ageWeightTem)
            fatTem = getRandNpcFatTem(ageWeightTem)
            randomNpcNewData = {
                "Name":randomNpcName,
                "Sex":randomNpcSex,
                "Age":randomNpcAgeTem,
                "Position":["0"],
                "AdvNpc":"1",
                "Weight":fatTem
            }
            CacheContorl.randomNpcList.append(randomNpcNewData)
        return CacheContorl.randomNpcList

sexWeightData = TextLoading.getTextData(TextLoading.temId,'RandomNpcSexWeight')
sexWeightMax = 0
for i in sexWeightData:
    sexWeightMax += int(sexWeightData[i])
sexWeightReginData = ValueHandle.getReginList(sexWeightData)
sexWeightReginList = ValueHandle.getListKeysIntList(list(sexWeightReginData.keys()))
# 按权重随机获取npc性别
def getRandNpcSex():
    nowWeight = random.randint(0,sexWeightMax - 1)
    weightRegin = next(x for x in sexWeightReginList if x > nowWeight)
    return sexWeightReginData[str(weightRegin)]

fatWeightData = TextLoading.getTextData(TextLoading.temId,'FatWeight')
# 按权重随机获取npc肥胖模板
def getRandNpcFatTem(agejudge):
    nowFatWeightData = fatWeightData[agejudge]
    nowFatTem = ValueHandle.getRandomForWeight(nowFatWeightData)
    return nowFatTem

ageTemWeightData = TextLoading.getTextData(TextLoading.temId,'AgeWeight')
# 按权重获取npc年龄模板
def getRandNpcAgeTem(agejudge):
    nowAgeWeightData  = ageTemWeightData[agejudge]
    nowAgeTem = ValueHandle.getRandomForWeight(nowAgeWeightData)
    return nowAgeTem

# 获取角色最大数量
def getCharacterIndexMax():
    playerData = CacheContorl.playObject['object']
    playerMax = len(playerData.keys()) - 1
    return playerMax

# 获取角色id列表
def getCharacterIdList():
    playerData = CacheContorl.playObject['object']
    playerList = ValueHandle.dictKeysToList(playerData)
    return playerList

# 初始化角色的位置
def initPlayerPosition():
    characterList = CacheContorl.npcTemData
    for i in range(0, len(characterList)):
        playerIdS = str(i + 1)
        characterData = characterList[i]
        characterInitPositionDirList = characterData['Position']
        characterInitPosition = MapHandle.getSceneIdForDirList(characterInitPositionDirList)
        characterPosition = CacheContorl.playObject['object'][playerIdS]['Position']
        MapHandle.playerMoveScene(characterPosition, characterInitPosition, playerIdS)
