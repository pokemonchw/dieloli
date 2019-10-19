import os,random,bisect
from script.Core import TextLoading,CacheContorl,GameConfig,GamePathConfig,ValueHandle,JsonHandle
from script.Design import ProportionalBar,AttrPrint

language = GameConfig.language
gamepath = GamePathConfig.gamepath

roleAttrPath = os.path.join(gamepath,'data',language,'RoleAttributes.json')
roleAttrData = JsonHandle._loadjson(roleAttrPath)
sexData = roleAttrData['Sex']
equipmentPath = os.path.join(gamepath,'data',language,'Equipment.json')
equipmentData = JsonHandle._loadjson(equipmentPath)
boysNameListData = TextLoading.getTextData(TextLoading.nameListPath,'Boys')
girlsNameListData = TextLoading.getTextData(TextLoading.nameListPath,'Girls')
familyNameListData = TextLoading.getTextData(TextLoading.familyNameListPath,'FamilyNameList')
sortFamilyIndex = sorted(familyNameListData.items(),key=lambda x:x[1])

familyRegionList = ValueHandle.getReginList(familyNameListData)
boysRegionList = ValueHandle.getReginList(boysNameListData)
girlsRegionList = ValueHandle.getReginList(girlsNameListData)

familyRegionIntList = list(map(int,familyRegionList))
boysRegionIntList = list(map(int,boysRegionList))
girlsRegionIntList = list(map(int,girlsRegionList))

def getSexExperienceText(sexExperienceData:dict,sexName:str) -> list:
    '''
    获取性经验描述文本
    Keyword arguments:
    sexExperienceData -- 性经验数据列表
    sexName -- 性别
    '''
    mouthExperience = TextLoading.getTextData(TextLoading.stageWordPath,'19') + str(sexExperienceData['mouthExperience'])
    bosomExperience = TextLoading.getTextData(TextLoading.stageWordPath,'20') + str(sexExperienceData['bosomExperience'])
    vaginaExperience = TextLoading.getTextData(TextLoading.stageWordPath,'21') + str(sexExperienceData['vaginaExperience'])
    clitorisExperience = TextLoading.getTextData(TextLoading.stageWordPath,'22') + str(sexExperienceData['clitorisExperience'])
    anusExperience = TextLoading.getTextData(TextLoading.stageWordPath,'23') + str(sexExperienceData['anusExperience'])
    penisExperience = TextLoading.getTextData(TextLoading.stageWordPath,'24') + str(sexExperienceData['penisExperience'])
    sexExperienceText = []
    sexList = list(sexData.keys())
    if sexName == sexList[0]:
        sexExperienceText = [mouthExperience,bosomExperience,anusExperience,penisExperience]
    elif sexName == sexList[1]:
        sexExperienceText = [mouthExperience,bosomExperience,vaginaExperience,clitorisExperience,anusExperience]
    elif sexName == sexList[2]:
        sexExperienceText = [mouthExperience,bosomExperience,vaginaExperience,clitorisExperience,anusExperience,penisExperience]
    elif sexName == sexList[3]:
        sexExperienceText = [mouthExperience,bosomExperience,anusExperience]
    return sexExperienceText

def getSexGradeTextList(sexGradeData:dict,sexName:str) -> list:
    '''
    获取性等级描述文本
    Keyword arguments:
    sexGradeData -- 性等级列表
    sexName -- 性别
    '''
    mouthText = TextLoading.getTextData(TextLoading.stageWordPath,'25') + getLevelTextColor(sexGradeData['mouthGrade'])
    bosomText = TextLoading.getTextData(TextLoading.stageWordPath,'26') + getLevelTextColor(sexGradeData['bosomGrade'])
    vaginaText = TextLoading.getTextData(TextLoading.stageWordPath,'27') + getLevelTextColor(sexGradeData['vaginaGrade'])
    clitorisText = TextLoading.getTextData(TextLoading.stageWordPath,'28') + getLevelTextColor(sexGradeData['clitorisGrade'])
    anusText = TextLoading.getTextData(TextLoading.stageWordPath,'29') + getLevelTextColor(sexGradeData['anusGrade'])
    penisText = TextLoading.getTextData(TextLoading.stageWordPath,'30') + getLevelTextColor(sexGradeData['penisGrade'])
    sexGradeTextList = []
    sexList = list(sexData.keys())
    if sexName == sexList[0]:
        sexGradeTextList = [mouthText,bosomText,anusText,penisText]
    elif sexName == sexList[1]:
        sexGradeTextList = [mouthText,bosomText,vaginaText,clitorisText,anusText]
    elif sexName == sexList[2]:
        sexGradeTextList = [mouthText,bosomText,vaginaText,clitorisText,anusText,penisText]
    elif sexName == sexList[3]:
        sexGradeTextList = [mouthText,bosomText,anusText]
    return sexGradeTextList

# 处理等级富文本
def getLevelTextColor(level:str) -> str:
    '''
    对等级文本进行富文本处理
    Keyword arguments:
    level -- 等级
    '''
    lowerLevel = level.lower()
    level = '<level' + lowerLevel + '>' + level + '</level' + lowerLevel + '>'
    return level

familyIndexMax = familyRegionIntList[len(familyRegionIntList) - 1]
def getRandomNameForSex(sexGrade:str) -> str:
    '''
    按性别随机生成姓名
    Keyword arguments:
    sexGrade -- 性别
    '''
    familyRandom = random.randint(1,familyIndexMax)
    familyRegionIndex = bisect.bisect_left(familyRegionIntList,familyRandom)
    familyRegion = familyRegionIntList[familyRegionIndex]
    familyName = familyRegionList[str(familyRegion)]
    if sexGrade == 'Man':
        sexJudge = 1
    elif sexGrade == 'Woman':
        sexJudge = 0
    else:
        sexJudge = random.randint(0,1)
    if sexJudge == 0:
        nameRandom = random.randint(1,girlsRegionIntList[-1])
        nameRegionIndex = bisect.bisect_left(girlsRegionIntList,nameRandom)
        nameRegion = girlsRegionIntList[nameRegionIndex]
        name = girlsRegionList[str(nameRegion)]
    else:
        nameRandom = random.randint(1,boysRegionIntList[-1])
        nameRegionIndex = bisect.bisect_left(boysRegionIntList,nameRandom)
        nameRegion = boysRegionIntList[nameRegionIndex]
        name = boysRegionList[str(nameRegion)]
    return familyName + name

def getSexText(sexId:str) -> str:
    '''
    获取性别对应文本
    Keyword arguments:
    sexId -- 性别
    '''
    data = roleAttrData['Sex']
    sexText = data[sexId]
    return sexText

def getClothingText(clothingList:dict) -> list:
    '''
    获取服装描述文本
    Keyword arguments:
    clothingList -- 服装数据
    '''
    coatid = int(clothingList["Coat"])
    pantsid = int(clothingList["Pants"])
    shoesid = int(clothingList["Shoes"])
    socksid = int(clothingList["Socks"])
    underwearid = int(clothingList["Underwear"])
    braid = int(clothingList["Bra"])
    underpantsid = int(clothingList["Underpants"])
    leggingsid = int(clothingList["Leggings"])
    clothingData = equipmentData["Clothing"]
    coatText = clothingData["Coat"][coatid]
    pantsText = clothingData["Pants"][pantsid]
    shoesText = clothingData["Shoes"][shoesid]
    socksText = clothingData["Socks"][socksid]
    underwearText = clothingData["Underwear"][underwearid]
    braText = clothingData["Bra"][braid]
    underpantsText = clothingData["Underpants"][underpantsid]
    leggingsText = clothingData["Leggings"][leggingsid]
    coatText = TextLoading.getTextData(TextLoading.stageWordPath,"41") + coatText
    pantsText = TextLoading.getTextData(TextLoading.stageWordPath, "42") + pantsText
    shoesText = TextLoading.getTextData(TextLoading.stageWordPath, "43") + shoesText
    socksText = TextLoading.getTextData(TextLoading.stageWordPath, "44") + socksText
    underwearText = TextLoading.getTextData(TextLoading.stageWordPath, "45") + underwearText
    braText = TextLoading.getTextData(TextLoading.stageWordPath, "46") + braText
    underpantsText = TextLoading.getTextData(TextLoading.stageWordPath, "47") + underpantsText
    leggingsText = TextLoading.getTextData(TextLoading.stageWordPath, "48") + leggingsText
    clothingTextList = [
        coatText,pantsText,shoesText,socksText,underwearText,
        braText,underpantsText,leggingsText
    ]
    return clothingTextList

def getGoldText(characterId:str) -> str:
    '''
    获取指定角色的金钱信息描述文本
    Keyword arguments:
    characterId -- 角色id
    '''
    goldData = CacheContorl.characterData['character'][characterId]['Gold']
    goldData = str(goldData)
    moneyText = TextLoading.getTextData(TextLoading.stageWordPath,'66')
    goldText = TextLoading.getTextData(TextLoading.stageWordPath,'67')
    goldText = goldText + goldData + moneyText
    return goldText

def getStateText(characterId:str) -> str:
    '''
    按角色Id获取状态描述信息
    Keyword arguments:
    characterId -- 角色Id
    '''
    state = CacheContorl.characterData['character'][characterId]['State']
    stateText = TextLoading.getTextData(TextLoading.stageWordPath,'132')[state]
    return TextLoading.getTextData(TextLoading.stageWordPath,'133') + stateText

def getStatureText(characterId:str) -> list:
    '''
    按角色Id获取身材描述信息
    Keyword arguments:
    characterId -- 角色Id
    '''
    characterData = CacheContorl.characterData['character'][characterId]
    return ''

def getCharacterAbbreviationsInfo(characterId:str) -> str:
    '''
    按角色id获取角色缩略信息文本
    Keyword arguments:
    characterId -- 角色id
    '''
    characterData = CacheContorl.characterData['character'][characterId]
    characterIdInfo = TextLoading.getTextData(TextLoading.stageWordPath, '0')
    characterIdText = characterIdInfo + characterId
    characterName = characterData['Name']
    characterSex = characterData['Sex']
    characterSexInfo = TextLoading.getTextData(TextLoading.stageWordPath, '2')
    characterSexTextData = TextLoading.getTextData(TextLoading.rolePath,'Sex')
    characterSexText = characterSexTextData[characterSex]
    characterSexText = characterSexInfo + characterSexText
    characterAge = characterData['Age']
    characterAgeInfo = TextLoading.getTextData(TextLoading.stageWordPath, '3')
    characterAgeText = characterAgeInfo + str(characterAge)
    characterHpAndMpText = AttrPrint.getHpAndMpText(characterId)
    characterIntimate = characterData['Intimate']
    characterIntimateInfo = TextLoading.getTextData(TextLoading.stageWordPath, '16')
    characterIntimateText = characterIntimateInfo + characterIntimate
    characterGraces = characterData['Graces']
    characterGracesInfo = TextLoading.getTextData(TextLoading.stageWordPath, '17')
    characterGracesText = characterGracesInfo + characterGraces
    abbreviationsInfo = characterIdText + ' ' + characterName + ' ' + characterSexText + ' ' + characterAgeText + ' ' + characterHpAndMpText + ' ' + characterIntimateText + ' ' + characterGracesText
    return abbreviationsInfo
