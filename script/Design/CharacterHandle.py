import random
import math
import uuid
import time
import itertools
import numpy
from script.Core import CacheContorl,ValueHandle,GameData,TextLoading,GamePathConfig,GameConfig
from script.Design import AttrCalculation,MapHandle,AttrText,Clothing,Nature,Character

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
    idList = iter([i+1 for i in range(len(CacheContorl.npcTemData))])
    npcDataIter = iter(CacheContorl.npcTemData)
    while(1):
        try:
            initCharacter(next(idList),next(npcDataIter))
        except StopIteration:
            break
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
        ageTem = AttrCalculation.judgeAgeGroup(characterData.Age)
        CacheContorl.TotalHeightByage.setdefault(ageTem,{})
        CacheContorl.TotalHeightByage[ageTem].setdefault(characterData.Sex,0)
        CacheContorl.TotalHeightByage[ageTem][characterData.Sex] += characterData.Height['NowHeight']
        CacheContorl.TotalNumberOfPeopleOfAllAges.setdefault(ageTem,{})
        CacheContorl.TotalNumberOfPeopleOfAllAges[ageTem].setdefault(characterData.Sex,0)
        CacheContorl.TotalNumberOfPeopleOfAllAges[ageTem][characterData.Sex] += 1
        CacheContorl.TotalBodyFatByage.setdefault(ageTem,{})
        CacheContorl.TotalBodyFatByage[ageTem].setdefault(characterData.Sex,0)
        CacheContorl.TotalBodyFatByage[ageTem][characterData.Sex] += characterData.BodyFat

def initCharacter(characterId:int,character:dict):
    '''
    按id生成角色属性
    Keyword arguments:
    characterId -- 角色id
    character -- 角色生成模板数据
    '''
    nowCharacter = Character.Character()
    nowCharacter.Name = character['Name']
    nowCharacter.Sex = character['Sex']
    if 'MotherTongue' in character:
        nowCharacter.MotherTongue = character['MotherTongue']
    if 'Age' in character:
        nowCharacter.Age = AttrCalculation.getAge(character['Age'])
    if 'Weight' in character:
        nowCharacter.WeigtTem = character['Weight']
    if 'BodyFat' in character:
        nowCharacter.BodyFatTem = character['BodyFat']
    else:
        nowCharacter.BodyFatTem = nowCharacter.WeigtTem
    nowCharacter.initAttr()
    CacheContorl.characterData['character'][characterId] = nowCharacter

def initCharacterTem():
    '''
    初始化角色模板数据
    '''
    npcData = getRandomNpcData()
    nowCharacterList = characterList.copy()
    npcData += [getDirCharacterTem(character) for character in nowCharacterList]
    numpy.random.shuffle(npcData)
    CacheContorl.npcTemData = npcData

def getDirCharacterTem(character:int) -> dict:
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
ageWeightMax = sum([int(ageWeightData[ageWeight]) for ageWeight in ageWeightData])
def getRandomNpcData() -> list:
    '''
    生成所有随机npc的数据模板
    '''
    if CacheContorl.randomNpcList == []:
        list(map(createRandomNpc,range(randomNpcMax)))
        return CacheContorl.randomNpcList

def createRandomNpc(id) -> dict:
    '''
    生成随机npc数据模板
    '''
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

def initCharacterDormitory():
    '''
    分配角色宿舍
    小于18岁，男生分配到男生宿舍，女生分配到女生宿舍，按宿舍楼层和角色年龄，从下往上，从小到大分配，其他性别分配到地下室，大于18岁，教师宿舍混居
    '''
    characterData = {}
    characterSexData = {
        "Man":{character:CacheContorl.characterData['character'][character].Age for character in CacheContorl.characterData['character'] if CacheContorl.characterData['character'][character].Age < 18 and CacheContorl.characterData['character'][character].Sex == 'Man'},
        "Woman":{character:CacheContorl.characterData['character'][character].Age for character in CacheContorl.characterData['character'] if CacheContorl.characterData['character'][character].Age < 18 and CacheContorl.characterData['character'][character].Sex == 'Woman'},
        "Other":{character:CacheContorl.characterData['character'][character].Age for character in CacheContorl.characterData['character'] if CacheContorl.characterData['character'][character].Age < 18 and CacheContorl.characterData['character'][character].Sex not in {"Man":0,"Woman":1}},
        "Teacher":{character:CacheContorl.characterData['character'][character].Age for character in CacheContorl.characterData['character'] if CacheContorl.characterData['character'][character].Age >= 18}
    }
    manMax = len(characterSexData['Man'])
    womanMax = len(characterSexData['Woman'])
    otherMax = len(characterSexData['Other'])
    teacherMax = len(characterSexData['Teacher'])
    characterSexData['Man'] = [k[0] for k in sorted(characterSexData['Man'].items(),key=lambda x:x[1])]
    characterSexData['Woman'] = [k[0] for k in sorted(characterSexData['Woman'].items(),key=lambda x:x[1])]
    characterSexData['Other'] = [k[0] for k in sorted(characterSexData['Other'].items(),key=lambda x:x[1])]
    characterSexData['Teacher'] = [k[0] for k in sorted(characterSexData['Teacher'].items(),key=lambda x:x[1])]
    teacherDormitory = {x:0 for x in sorted(CacheContorl.placeData['TeacherDormitory'],key=lambda x:x[0])}
    maleDormitory = {key:CacheContorl.placeData[key] for key in CacheContorl.placeData if 'MaleDormitory' in key}
    femaleDormitory = {key:CacheContorl.placeData[key] for key in CacheContorl.placeData if 'FemaleDormitory' in key}
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
    manList = characterSexData['Man'].copy()
    for character in characterSexData['Man']:
        nowRoom = list(maleDormitory.keys())[0]
        CacheContorl.characterData['character'][character].Dormitory = nowRoom
        maleDormitory[nowRoom] += 1
        if maleDormitory[nowRoom] >= singleRoomMan:
            del maleDormitory[nowRoom]
    for character in characterSexData['Woman']:
        nowRoom = list(femaleDormitory.keys())[0]
        CacheContorl.characterData['character'][character].Dormitory = nowRoom
        femaleDormitory[nowRoom] += 1
        if femaleDormitory[nowRoom] >= singleRoomWoman:
            del femaleDormitory[nowRoom]
    for character in characterSexData['Other']:
        nowRoom = list(basement.keys())[0]
        CacheContorl.characterData['character'][character].Dormitory = nowRoom
        basement[nowRoom] += 1
        if basement[nowRoom] >= singleRoomBasement:
            del basement[nowRoom]
    for character in characterSexData['Teacher']:
        nowRoom = list(teacherDormitory.keys())[0]
        CacheContorl.characterData['character'][character].Dormitory = nowRoom
        teacherDormitory[nowRoom] += 1
        if teacherDormitory[nowRoom] >= singleRoomTeacher:
            del teacherDormitory[nowRoom]

def initCharacterPosition():
    '''
    初始化角色位置
    '''
    for character in CacheContorl.characterData['character']:
        characterPosition = CacheContorl.characterData['character'][character].Position
        characterDormitory = CacheContorl.characterData['character'][character].Dormitory
        characterDormitory = MapHandle.getMapSystemPathForStr(characterDormitory)
        MapHandle.characterMoveScene(characterPosition,characterDormitory,character)
