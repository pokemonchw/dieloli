import os,random
from script.Core import CacheContorl,ValueHandle,GameData,TextLoading,GamePathConfig,GameConfig
from script.Design import AttrCalculation,MapHandle,AttrText

language = GameConfig.language
gamepath = GamePathConfig.gamepath
featuresList = AttrCalculation.getFeaturesList()
sexList = list(TextLoading.getTextData(TextLoading.roleId, 'Sex'))
ageTemList = list(TextLoading.getTextData(TextLoading.temId,'AgeTem'))

# 初始化角色数据
def initCharacterList():
    initCharacterTem()
    characterList = CacheContorl.npcTemData
    i = 0
    for character in characterList:
        AttrCalculation.initTemporaryObject()
        playerId = str(i + 1)
        i += 1
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
            if ageTem == 'SchoolAgeChild':
                if characterSex == sexList[0]:
                    CacheContorl.featuresList['Age'] = featuresList["Age"][0]
                elif characterSex == sexList[1]:
                    CacheContorl.featuresList['Age'] = featuresList["Age"][1]
                else:
                    CacheContorl.featuresList['Age'] = featuresList["Age"][2]
            elif ageTem == 'OldAdult':
                CacheContorl.featuresList['Age'] = featuresList["Age"][3]
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
        weight = AttrCalculation.getWeight(weightTemName, height['NowHeight'])
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
    initPlayerPosition()

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

# 获取随机npc数据
def getRandomNpcData():
    if CacheContorl.randomNpcList == []:
        randomNpcMax = int(GameConfig.random_npc_max)
        randomTeacherProportion = int(GameConfig.proportion_teacher)
        randomStudentProportion = int(GameConfig.proportion_student)
        randomTeacherMax = round(randomNpcMax * (randomTeacherProportion / 100))
        randomStudentMax = round(randomNpcMax * (randomStudentProportion / 100))
        teacherIndex = 0
        randomNpcMax = randomStudentMax + randomTeacherMax
        for i in range(0,randomNpcMax):
            randomNpcSex = random.choice(sexList)
            randomNpcName = AttrText.getRandomNameForSex(randomNpcSex)
            if teacherIndex < randomTeacherMax:
                teacherIndex += 1
                teacherAgeTem = ageTemList[:2]
                randomNpcAgeTem = random.choice(teacherAgeTem)
            else:
                studentAgeTem = ageTemList[3:]
                randomNpcAgeTem = random.choice(studentAgeTem)
            randomNpcNewData = {
                "Name":randomNpcName,
                "Sex":randomNpcSex,
                "Age":randomNpcAgeTem,
                "Position":["0"],
                "AdvNpc":"1"
            }
            CacheContorl.randomNpcList.append(randomNpcNewData)
        return CacheContorl.randomNpcList

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
