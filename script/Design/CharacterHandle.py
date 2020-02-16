import random
import math
import uuid
import time
from script.Core import CacheContorl,ValueHandle,GameData,TextLoading,GamePathConfig,GameConfig
from script.Design import AttrCalculation,MapHandle,AttrText,Clothing,Nature

language = GameConfig.language
gamepath = GamePathConfig.gamepath
sexList = list(TextLoading.getTextData(TextLoading.rolePath, 'Sex'))
ageTemList = list(TextLoading.getTextData(TextLoading.attrTemplatePath,'AgeTem'))
characterList = list(GameData.gamedata[language]['character'].keys())

def initCharacterList():
    '''
    初始生成所有npc数据
    '''
    initCharacterTem()
    t1 = time.time()
    i = 1
    for character in CacheContorl.npcTemData:
        initCharacter((i,character))
        i += 1
    t2 = time.time()
    print(t2-t1)
    indexCharacterAverageValue()
    calculateTheAverageValueOfEachAttributeOfEachAgeGroup()

def calculateTheAverageValueOfEachAttributeOfEachAgeGroup():
    '''
    计算各年龄段各项属性平均值
    '''
    CacheContorl.AverageBodyFatByage = {sex:{ageTem:CacheContorl.TotalBodyFatByage[sex][ageTem] / CacheContorl.TotalNumberOfPeopleOfAllAges[sex][ageTem] for ageTem in CacheContorl.TotalBodyFatByage[sex]} for sex in CacheContorl.TotalBodyFatByage}
    CacheContorl.AverageHeightByage = {sex:{ageTem:CacheContorl.TotalHeightByage[sex][ageTem] / CacheContorl.TotalNumberOfPeopleOfAllAges[sex][ageTem] for ageTem in CacheContorl.TotalHeightByage[sex]} for sex in CacheContorl.TotalHeightByage}

def indexCharacterAverageValue():
    '''
    统计各年龄段所有角色各属性总值
    '''
    for character in CacheContorl.characterData['character']:
        characterData = CacheContorl.characterData['character'][character]
        ageTem = AttrCalculation.judgeAgeGroup(characterData['Age'])
        CacheContorl.TotalHeightByage.setdefault(ageTem,{})
        CacheContorl.TotalHeightByage[ageTem].setdefault(characterData['Sex'],0)
        CacheContorl.TotalHeightByage[ageTem][characterData['Sex']] += characterData['Height']['NowHeight']
        CacheContorl.TotalNumberOfPeopleOfAllAges.setdefault(ageTem,{})
        CacheContorl.TotalNumberOfPeopleOfAllAges[ageTem].setdefault(characterData['Sex'],0)
        CacheContorl.TotalNumberOfPeopleOfAllAges[ageTem][characterData['Sex']] += 1
        CacheContorl.TotalBodyFatByage.setdefault(ageTem,{})
        CacheContorl.TotalBodyFatByage[ageTem].setdefault(characterData['Sex'],0)
        CacheContorl.TotalBodyFatByage[ageTem][characterData['Sex']] += characterData['BodyFat']

def initCharacter(*args):
    '''
    按id生成角色属性
    Keyword arguments:
    nowId -- 角色id
    character -- 角色生成模板数据
    '''
    args = args[0]
    characterId = str(args[0])
    character = args[1]
    CacheContorl.characterData['character'][characterId] = AttrCalculation.initTemporaryCharacter()
    characterName = character['Name']
    characterSex = character['Sex']
    CacheContorl.characterData['character'][characterId]['Sex'] = characterSex
    defaultAttr = AttrCalculation.getAttr(characterSex)
    defaultAttr['Name'] = characterName
    defaultAttr['Sex'] = characterSex
    if 'MotherTongue' in character:
        defaultAttr['Language'][character['MotherTongue']] = 10000
        defaultAttr['MotherTongue'] = character['MotherTongue']
    else:
        defaultAttr['Language']['Chinese'] = 10000
    if 'Age' in character:
        ageTem = character['Age']
        characterAge = AttrCalculation.getAge(ageTem)
        defaultAttr['Age'] = characterAge
        defaultAttr['Birthday'] = AttrCalculation.getRandNpcBirthDay(characterAge)
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
    defaultAttr['BodyFat'] = bodyFat
    measurements = AttrCalculation.getMeasurements(characterSex, height['NowHeight'], weight,bodyFat,bodyFatTem)
    defaultAttr['Measirements'] = measurements
    defaultAttr['Knowledge'] = {}
    if "SexExperience" in character:
        sexExperienceTem = character['SexExperience']
    else:
        sexExperienceTem = getRandNpcSexExperienceTem(defaultAttr['Age'],defaultAttr['Sex'])
    defaultAttr['SexExperience'] = AttrCalculation.getSexExperience(sexExperienceTem)
    defaultAttr['SexGrade'] = AttrCalculation.getSexGrade(defaultAttr['SexExperience'])
    if 'Clothing' in character:
        clothingTem = character['Clothing']
    else:
        clothingTem = 'Uniform'
    defaultClothingData = Clothing.creatorSuit(clothingTem,characterSex)
    for clothing in defaultClothingData:
        defaultAttr['Clothing'][clothing][uuid.uuid1()] = defaultClothingData[clothing]
    if 'Chest' in character:
        chest = AttrCalculation.getChest(chestTem,defaultAttr['Birthday'])
        defaultAttr['Chest'] = chest
    CacheContorl.characterData['character'][characterId].update(defaultAttr)
    Clothing.characterPutOnClothing(characterId)
    Nature.initCharacterNature(characterId)

def initCharacterTem():
    '''
    初始化角色模板数据
    '''
    npcData = getRandomNpcData()
    nowCharacterList = characterList.copy()
    npcData += [getDirCharacterTem(character) for character in nowCharacterList]
    CacheContorl.npcTemData = npcData

def getDirCharacterTem(character:str) -> dict:
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
def getRandomNpcData() -> list:
    '''
    生成所有随机npc的数据模板
    '''
    if CacheContorl.randomNpcList == []:
        ageWeightMax = sum([int(ageWeightData[ageWeight]) for ageWeight in ageWeightData])
        for i in range(0,randomNpcMax):
            nowAgeWeight = random.randint(-1,ageWeightMax - 1)
            nowAgeWeightRegin = ValueHandle.getNextValueForList(nowAgeWeight,ageWeightReginList)
            ageWeightTem = ageWeightReginData[nowAgeWeightRegin]
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
sexWeightMax = sum([int(sexWeightData[weight]) for weight in sexWeightData])
sexWeightReginData = ValueHandle.getReginList(sexWeightData)
sexWeightReginList = list(map(int,sexWeightReginData.keys()))
def getRandNpcSex() -> str:
    '''
    随机获取npc性别
    '''
    nowWeight = random.randint(0,sexWeightMax - 1)
    weightRegin = ValueHandle.getNextValueForList(nowWeight,sexWeightReginList)
    return sexWeightReginData[weightRegin]

fatWeightData = TextLoading.getTextData(TextLoading.attrTemplatePath,'FatWeight')
def getRandNpcFatTem(agejudge:str) -> str:
    '''
    按人群年龄段体重分布比例随机生成体重模板
    Keyword arguments:
    agejudge -- 年龄段
    '''
    nowFatWeightData = fatWeightData[agejudge]
    nowFatTem = ValueHandle.getRandomForWeight(nowFatWeightData)
    return nowFatTem

def getRandNpcSexExperienceTem(age:int,sex:str) -> str:
    '''
    按年龄范围随机获取性经验模板
    Keyword arguments:
    age -- 年龄
    sex -- 性别
    '''
    ageJudgeSexExperienceTemData = TextLoading.getTextData(TextLoading.attrTemplatePath,'AgeJudgeSexExperienceTem')
    if sex == 'Asexual':
        sex = 'Woman'
    if sex == 'Futa':
        sex = 'Man'
    nowTemData = ageJudgeSexExperienceTemData[sex]
    ageRegionList = [int(i) for i in nowTemData.keys()]
    ageRegion = str(ValueHandle.getOldValueForList(age,ageRegionList))
    ageRegionData = nowTemData[ageRegion]
    return ValueHandle.getRandomForWeight(ageRegionData)

bodyFatWeightData = TextLoading.getTextData(TextLoading.attrTemplatePath,'BodyFatWeight')
def getRandNpcBodyFatTem(ageJudge:str,bmiTem:str) -> str:
    '''
    按年龄段体脂率分布比例随机生成体脂率模板
    Keyword arguments:
    ageJudge -- 年龄段
    bmiTem -- bmi模板
    '''
    nowBodyFatData = bodyFatWeightData[ageJudge][bmiTem]
    return ValueHandle.getRandomForWeight(nowBodyFatData)

ageTemWeightData = TextLoading.getTextData(TextLoading.attrTemplatePath,'AgeWeight')
def getRandNpcAgeTem(agejudge:str) -> int:
    '''
    按年龄断随机生成npc年龄
    Keyword arguments:
    ageJudge -- 年龄段
    '''
    nowAgeWeightData  = ageTemWeightData[agejudge]
    nowAgeTem = ValueHandle.getRandomForWeight(nowAgeWeightData)
    return nowAgeTem

def getCharacterIndexMax() -> int:
    '''
    获取角色数量
    '''
    characterData = CacheContorl.characterData['character']
    characterDataMax = len(characterData.keys()) - 1
    return characterDataMax

def getCharacterIdList() -> list:
    '''
    获取角色id列表
    '''
    characterData = CacheContorl.characterData['character']
    return list(characterData.keys())

def initCharacterDormitory():
    '''
    分配角色宿舍
    小于18岁，男生分配到男生宿舍，女生分配到女生宿舍，按宿舍楼层和角色年龄，从下往上，从小到大分配，其他性别分配到地下室，大于18岁，教师宿舍混居
    '''
    characterData = {}
    characterSexData = {
        "Man":{},
        "Woman":{},
        "Other":{},
        "Teacher":{}
    }
    for character in CacheContorl.characterData['character']:
        if CacheContorl.characterData['character'][character]['Age'] < 18:
            if CacheContorl.characterData['character'][character]['Sex'] in ['Man','Woman']:
                nowSex = CacheContorl.characterData['character'][character]['Sex']
                characterSexData[nowSex][character] = CacheContorl.characterData['character'][character]['Age']
            else:
                characterSexData['Other'][character] = CacheContorl.characterData['character'][character]['Age']
        else:
            characterSexData['Teacher'][character] = CacheContorl.characterData['character'][character]['Age']
    manMax = len(characterSexData['Man'])
    womanMax = len(characterSexData['Woman'])
    otherMax = len(characterSexData['Other'])
    teacherMax = len(characterSexData['Teacher'])
    characterSexData['Man'] = [k[0] for k in sorted(characterSexData['Man'].items(),key=lambda x:x[1])]
    characterSexData['Woman'] = [k[0] for k in sorted(characterSexData['Woman'].items(),key=lambda x:x[1])]
    characterSexData['Other'] = [k[0] for k in sorted(characterSexData['Other'].items(),key=lambda x:x[1])]
    characterSexData['Teacher'] = [k[0] for k in sorted(characterSexData['Teacher'].items(),key=lambda x:x[1])]
    teacherDormitory = {x:0 for x in sorted(CacheContorl.placeData['TeacherDormitory'],key=lambda x:x[0])}
    maleDormitory = {}
    femaleDormitory = {}
    for key in CacheContorl.placeData:
        if 'FemaleDormitory' in key:
            femaleDormitory[key] = CacheContorl.placeData[key]
        elif 'MaleDormitory' in key:
            maleDormitory[key] = CacheContorl.placeData[key]
    maleDormitory = {x:0 for j in [k[1] for k in sorted(maleDormitory.items(),key=lambda x:x[0])] for x in j}
    femaleDormitory = {x:0 for j in [k[1] for k in sorted(femaleDormitory.items(),key=lambda x:x[0])] for x in j}
    basement = {x:0 for x in CacheContorl.placeData['Basement']}
    maleDormitoryMax = len(maleDormitory.keys())
    femaleDormitoryMax = len(femaleDormitory.keys())
    teacherDormitoryMax = len(teacherDormitory)
    basementMax = len(basement)
    singleRoomMan = math.ceil(manMax / maleDormitoryMax)
    singleRoomWoman = math.ceil(womanMax / femaleDormitoryMax)
    singleRoomBasement = math.ceil(otherMax / basementMax)
    singleRoomTeacher = math.ceil(teacherMax / teacherDormitoryMax)
    for character in characterSexData['Man']:
        nowRoom = list(maleDormitory.keys())[0]
        CacheContorl.characterData['character'][character]['Dormitory'] = nowRoom
        maleDormitory[nowRoom] += 1
        if maleDormitory[nowRoom] >= singleRoomMan:
            del maleDormitory[nowRoom]
    for character in characterSexData['Woman']:
        nowRoom = list(femaleDormitory.keys())[0]
        CacheContorl.characterData['character'][character]['Dormitory'] = nowRoom
        femaleDormitory[nowRoom] += 1
        if femaleDormitory[nowRoom] >= singleRoomWoman:
            del femaleDormitory[nowRoom]
    for character in characterSexData['Other']:
        nowRoom = list(basement.keys())[0]
        CacheContorl.characterData['character'][character]['Dormitory'] = nowRoom
        basement[nowRoom] += 1
        if basement[nowRoom] >= singleRoomBasement:
            del basement[nowRoom]
    for character in characterSexData['Teacher']:
        nowRoom = list(teacherDormitory.keys())[0]
        CacheContorl.characterData['character'][character]['Dormitory'] = nowRoom
        teacherDormitory[nowRoom] += 1
        if teacherDormitory[nowRoom] >= singleRoomTeacher:
            del teacherDormitory[nowRoom]

def initCharacterPosition():
    '''
    初始化角色位置
    '''
    for character in CacheContorl.characterData['character']:
        characterPosition = CacheContorl.characterData['character'][character]['Position']
        characterDormitory = CacheContorl.characterData['character'][character]['Dormitory']
        characterDormitory = MapHandle.getMapSystemPathForStr(characterDormitory)
        MapHandle.characterMoveScene(characterPosition,characterDormitory,character)
