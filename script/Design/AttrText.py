import os,random,bisect
from script.Core import TextLoading,CacheContorl,GameConfig,GamePathConfig,ValueHandle,JsonHandle
from script.Design import ProportionalBar,AttrPrint,AttrCalculation,MapHandle

language = GameConfig.language
gamepath = GamePathConfig.gamepath

roleAttrPath = os.path.join(gamepath,'data',language,'RoleAttributes.json')
roleAttrData = JsonHandle._loadjson(roleAttrPath)
sexData = roleAttrData['Sex']
equipmentPath = os.path.join(gamepath,'data',language,'Equipment.json')
equipmentData = JsonHandle._loadjson(equipmentPath)

def getSexExperienceText(sexExperienceData:dict,sexName:str) -> list:
    '''
    获取性经验描述文本
    Keyword arguments:
    sexExperienceData -- 性经验数据列表
    sexName -- 性别
    '''
    mouthExperience = TextLoading.getTextData(TextLoading.stageWordPath,'19') + f"{sexExperienceData['mouthExperience']}"
    bosomExperience = TextLoading.getTextData(TextLoading.stageWordPath,'20') + f"{sexExperienceData['bosomExperience']}"
    vaginaExperience = TextLoading.getTextData(TextLoading.stageWordPath,'21') + f"{sexExperienceData['vaginaExperience']}"
    clitorisExperience = TextLoading.getTextData(TextLoading.stageWordPath,'22') + f"{sexExperienceData['clitorisExperience']}"
    anusExperience = TextLoading.getTextData(TextLoading.stageWordPath,'23') + f"{sexExperienceData['anusExperience']}"
    penisExperience = TextLoading.getTextData(TextLoading.stageWordPath,'24') + f"{sexExperienceData['penisExperience']}"
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
    return f'<level{lowerLevel}>{level}</level{lowerLevel}>'

def getRandomNameForSex(sexGrade:str) -> str:
    '''
    按性别随机生成姓名
    Keyword arguments:
    sexGrade -- 性别
    '''
    familyRandom = random.randint(1,CacheContorl.familyRegionIntList[-1])
    familyRegionIndex = bisect.bisect_left(CacheContorl.familyRegionIntList,familyRandom)
    familyRegion = CacheContorl.familyRegionIntList[familyRegionIndex]
    familyName = CacheContorl.familyRegionList[familyRegion]
    if sexGrade == 'Man':
        sexJudge = 1
    elif sexGrade == 'Woman':
        sexJudge = 0
    else:
        sexJudge = random.randint(0,1)
    if sexJudge == 0:
        nameRandom = random.randint(1,CacheContorl.girlsRegionIntList[-1])
        nameRegionIndex = bisect.bisect_left(CacheContorl.girlsRegionIntList,nameRandom)
        nameRegion = CacheContorl.girlsRegionIntList[nameRegionIndex]
        name = CacheContorl.girlsRegionList[nameRegion]
    else:
        nameRandom = random.randint(1,CacheContorl.boysRegionIntList[-2])
        nameRegionIndex = bisect.bisect_left(CacheContorl.boysRegionIntList,nameRandom)
        nameRegion = CacheContorl.boysRegionIntList[nameRegionIndex]
        name = CacheContorl.boysRegionList[nameRegion]
    return familyName + name

def getSeeAttrPanelHeadCharacterInfo(characterId:int) -> str:
    '''
    获取查看角色属性面板头部角色缩略信息文本
    Keyword arguments:
    characterId -- 角色Id
    '''
    characterData = CacheContorl.characterData['character'][characterId]
    characterIdText = TextLoading.getTextData(TextLoading.stageWordPath, '0') + f'{characterId}'
    name = characterData.Name
    nickName = characterData.NickName
    characterName = TextLoading.getTextData(TextLoading.stageWordPath,'13') + name
    characterNickName = TextLoading.getTextData(TextLoading.stageWordPath,'12') + nickName
    sex = characterData.Sex
    sexText = TextLoading.getTextData(TextLoading.stageWordPath, '2') + getSexText(sex)
    nameText = characterIdText + ' ' + characterName + ' ' + characterNickName + ' ' + sexText
    return nameText

def getSexText(sexId:str) -> str:
    '''
    获取性别对应文本
    Keyword arguments:
    sexId -- 性别
    '''
    return roleAttrData['Sex'][sexId]

def getEngravingText(eList:dict) -> list:
    '''
    获取刻印描述文本
    Keyword arguments:
    eList -- 刻印数据
    '''
    painLevel = str(eList["Pain"])
    happyLevel = str(eList["Happy"])
    yieldLevel = str(eList["Yield"])
    fearLevel = str(eList["Fear"])
    resistanceLevel = str(eList["Resistance"])
    painLevelFix = TextLoading.getTextData(TextLoading.stageWordPath,'31')
    happyLevelFix = TextLoading.getTextData(TextLoading.stageWordPath,'32')
    yieldLevelFix = TextLoading.getTextData(TextLoading.stageWordPath,'33')
    fearLevelFix = TextLoading.getTextData(TextLoading.stageWordPath,'34')
    resistanceLevelFix = TextLoading.getTextData(TextLoading.stageWordPath,'35')
    LVText = TextLoading.getTextData(TextLoading.stageWordPath,'36')
    levelList = [painLevel,happyLevel,yieldLevel,fearLevel,resistanceLevel]
    levelFixList = [painLevelFix,happyLevelFix,yieldLevelFix,fearLevelFix,resistanceLevelFix]
    levelTextList = [f'{levelFixList[i]}{LVText}{levelList[i]}' for i in range(len(levelList))]
    levelBarList = [f"{ProportionalBar.getCountBar(levelTextList[i],3,levelList[i],'engravingemptybar')}" for i in range(len(levelList))]
    return levelBarList

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
    goldData = CacheContorl.characterData['character'][characterId].Gold
    goldData = str(goldData)
    moneyText = TextLoading.getTextData(TextLoading.stageWordPath,'66')
    goldText = TextLoading.getTextData(TextLoading.stageWordPath,'67')
    goldText = goldText + goldData + moneyText
    return goldText

def getLevelColorText(exp):
    '''
    计算经验对应等级并获取富文本
    Keyword arguments:
    exp -- 经验
    '''
    return getLevelTextColor(AttrCalculation.judgeGrade(exp))

def getStateText(characterId:str) -> str:
    '''
    按角色Id获取状态描述信息
    Keyword arguments:
    characterId -- 角色Id
    '''
    state = CacheContorl.characterData['character'][characterId].State
    stateText = TextLoading.getTextData(TextLoading.stageWordPath,'132')[state]
    return TextLoading.getTextData(TextLoading.stageWordPath,'133') + stateText

def judgeCharacterStature(characterId):
    '''
    校验身材信息
    Keyword arguments:
    characterId -- 角色Id
    '''
    characterData = CacheContorl.characterData['character'][characterId]
    selfSex = CacheContorl.characterData['character'][0].Sex
    targetSex = characterData.Sex
    ageJudge = 'Similar'
    selfAge = CacheContorl.characterData['character'][0].Age
    targetAge = characterData.Age
    ageDisparity = selfAge - targetAge
    if ageDisparity < -2 and ageDisparity >= -5:
        ageJudge = 'SlightlyHigh'
    elif ageDisparity < -5 and ageDisparity >= -15:
        ageJudge = 'High'
    elif ageDisparity < -15 and ageDisparity >= -30:
        ageJudge = 'VeryHigh'
    elif ageDisparity < -30 and ageDisparity >= -60:
        ageJudge = 'SuperHigh'
    elif ageDisparity < -60:
        ageJudge = 'ExtremelyHigh'
    elif ageDisparity > 2 and ageDisparity <= 5:
        ageJudge = 'SlightlyLow'
    elif ageDisparity >5 and ageDisparity <= 15:
        ageJudge = 'Low'
    elif ageDisparity > 15 and ageDisparity <= 30:
        ageJudge = 'VeryLow'
    elif ageDisparity > 30 and ageDisparity <= 60:
        ageJudge = 'SuperLow'
    elif ageDisparity > 60:
        ageJudge = 'ExtremelyLow'
    bodyFat = characterData.BodyFat
    ageTem = AttrCalculation.judgeAgeGroup(targetAge)
    averageBodyFat = CacheContorl.AverageBodyFatByage[ageTem][targetSex]
    bodyFatJudge = 'Similar'
    if bodyFat < averageBodyFat * 1.15 and bodyFat >= averageBodyFat * 1.05:
        bodyFatJudge = 'SlilghtlyHeight'
    elif bodyFat < averageBodyFat * 1.25 and bodyFat >= averageBodyFat * 1.15:
        bodyFatJudge = 'Height'
    elif bodyFat < averageBodyFat * 1.35 and bodyFat >= averageBodyFat * 1.25:
        bodyFatJudge = 'VeryHeight'
    elif bodyFat < averageBodyFat * 1.45 and bodyFat >= averageBodyFat * 1.35:
        bodyFatJudge = 'SuperHeight'
    elif bodyFat > averageBodyFat * 1.45:
        bodyFatJudge = 'ExtremelyHeight'
    elif bodyFat < averageBodyFat * 0.95 and bodyFat >= averageBodyFat * 0.85:
        bodyFatJudge = 'SlilghtlyLow'
    elif bodyFat < averageBodyFat * 0.85 and bodyFat >= averageBodyFat * 0.75:
        bodyFatJudge = 'Low'
    elif bodyFat < averageBodyFat * 0.75 and bodyFat >= averageBodyFat * 0.65:
        bodyFatJudge = 'VeryLow'
    elif bodyFat < averageBodyFat * 0.65 and bodyFat >= averageBodyFat * 0.55:
        bodyFatJudge = 'SuperLow'
    elif bodyFat < averageBodyFat * 0.55:
        bodyFatJudge = 'ExtremelyLow'
    averageHeight = CacheContorl.AverageHeightByage[ageTem][targetSex]
    height = characterData.Height['NowHeight']
    heightJudge = 'Similar'
    if height < averageHeight * 1.15 and height >= averageHeight * 1.05:
        heightJudge = 'SlilghtlyHeight'
    elif height < averageHeight * 1.25 and height >= averageHeight * 1.15:
        heightJudge = 'Height'
    elif height < averageHeight * 1.35 and height >= averageHeight * 1.25:
        heightJudge = 'VeryHeight'
    elif height < averageHeight * 1.45 and height >= averageHeight * 1.35:
        heightJudge = 'SuperHeight'
    elif height > averageHeight * 1.45:
        heightJudge = 'ExtremelyHeight'
    elif height < averageHeight * 0.95 and height >= averageHeight * 0.85:
        heightJudge = 'SlilghtlyLow'
    elif height < averageHeight * 0.85 and height >= averageHeight * 0.75:
        heightJudge = 'Low'
    elif height < averageHeight * 0.75 and height >= averageHeight * 0.65:
        heightJudge = 'VeryLow'
    elif height < averageHeight * 0.65 and height >= averageHeight * 0.55:
        heightJudge = 'SuperLow'
    elif height < averageHeight:
        heightJudge = 'ExtremelyLow'
    playerBodyFat = characterData.BodyFat
    playerBodyFatJudge = 'Similar'
    if bodyFat < playerBodyFat * 1.15 and bodyFat >= playerBodyFat * 1.05:
        playerBodyFatJudge = 'SlilghtlyHeight'
    elif bodyFat < playerBodyFat * 1.25 and bodyFat >= playerBodyFat * 1.15:
        playerBodyFatJudge = 'Height'
    elif bodyFat < playerBodyFat * 1.35 and bodyFat >= playerBodyFat * 1.25:
        playerBodyFatJudge = 'VeryHeight'
    elif bodyFat < playerBodyFat * 1.45 and bodyFat >= playerBodyFat * 1.35:
        playerBodyFatJudge = 'SuperHeight'
    elif bodyFat > playerBodyFat * 1.45:
        playerBodyFatJudge = 'ExtremelyHeight'
    elif bodyFat < playerBodyFat * 0.95 and bodyFat >= playerBodyFat * 0.85:
        playerBodyFatJudge = 'SlilghtlyLow'
    elif bodyFat < playerBodyFat * 0.85 and bodyFat >= playerBodyFat * 0.75:
        playerBodyFatJudge = 'Low'
    elif bodyFat < playerBodyFat * 0.75 and bodyFat >= playerBodyFat * 0.65:
        playerBodyFatJudge = 'VeryLow'
    elif bodyFat < playerBodyFat * 0.65 and bodyFat >= playerBodyFat * 0.55:
        playerBodyFatJudge = 'SuperLow'
    elif bodyFat < playerBodyFat * 0.55:
        playerBodyFatJudge = 'ExtremelyLow'
    playerHeight = CacheContorl.characterData['character'][0].Height['NowHeight']
    playerHeightJudge = 'Similar'
    if height < playerHeight * 1.15 and height >= playerHeight * 1.05:
        playerHeightJudge = 'SlilghtlyHeight'
    elif height < playerHeight * 1.25 and height >= playerHeight * 1.15:
        playerHeightJudge = 'Height'
    elif height < playerHeight * 1.35 and height >= playerHeight * 1.25:
        playerHeightJudge = 'VeryHeight'
    elif height < playerHeight * 1.45 and height >= playerHeight * 1.35:
        playerHeightJudge = 'SuperHeight'
    elif height > playerHeight * 1.45:
        playerHeightJudge = 'ExtremelyHeight'
    elif height < playerHeight * 0.95 and height >= playerHeight * 0.85:
        playerHeightJudge = 'SlilghtlyLow'
    elif height < playerHeight * 0.85 and height >= playerHeight * 0.75:
        playerHeightJudge = 'Low'
    elif height < playerHeight * 0.75 and height >= playerHeight * 0.65:
        playerHeightJudge = 'VeryLow'
    elif height < playerHeight * 0.65 and height >= playerHeight * 0.55:
        playerHeightJudge = 'SuperLow'
    elif height < playerHeight * 0.55:
        playerHeightJudge = 'ExtremelyLow'
    playerSex = CacheContorl.characterData['character'][0].Sex
    playerAge = CacheContorl.characterData['character'][0].Age
    playerAgeTem = AttrCalculation.judgeAgeGroup(CacheContorl.characterData['character'][0].Age)
    averageBodyFat = CacheContorl.AverageBodyFatByage[playerAgeTem][playerSex]
    averageHeight = CacheContorl.AverageHeightByage[playerAgeTem][playerSex]
    playerAverageBodyFatJudge = 'Similar'
    if playerBodyFat < averageBodyFat * 1.15 and playerBodyFat >= averageBodyFat * 1.05:
        playerAverageBodyFatJudge = 'SlilghtlyHeight'
    elif playerBodyFat < averageBodyFat * 1.25 and playerBodyFat >= averageBodyFat * 1.15:
        playerAverageBodyFatJudge = 'Height'
    elif playerBodyFat < averageBodyFat * 1.35 and playerBodyFat >= averageBodyFat * 1.25:
        playerAverageBodyFatJudge = 'VeryHeight'
    elif playerBodyFat < averageBodyFat * 1.45 and playerBodyFat >= averageBodyFat * 1.35:
        playerAverageBodyFatJudge = 'SuperHeight'
    elif playerBodyFat > averageBodyFat * 1.45:
        playerAverageBodyFatJudge = 'ExtremelyHeight'
    elif playerBodyFat < averageBodyFat * 0.95 and playerBodyFat >= averageBodyFat * 0.85:
        playerAverageBodyFatJudge = 'SlilghtlyLow'
    elif playerBodyFat < averageBodyFat * 0.85 and playerBodyFat >= averageBodyFat * 0.75:
        playerAverageBodyFatJudge = 'Low'
    elif playerBodyFat < averageBodyFat * 0.75 and playerBodyFat >= averageBodyFat * 0.65:
        playerAverageBodyFatJudge = 'VeryLow'
    elif playerBodyFat < averageBodyFat * 0.65 and playerBodyFat >= averageBodyFat * 0.55:
        playerAverageBodyFatJudge = 'SuperLow'
    elif playerBodyFat < averageBodyFat * 0.55:
        playerAverageBodyFatJudge = 'ExtremelyLow'
    playerAverageHeightJudge = 'Similar'
    if playerHeight < averageHeight * 1.15 and playerHeight >= averageHeight * 1.05:
        playerAverageHeightJudge = 'SlilghtlyHeight'
    elif playerHeight < averageHeight * 1.25 and playerHeight >= averageHeight * 1.15:
        playerAverageHeightJudge = 'Height'
    elif playerHeight < averageHeight * 1.35 and playerHeight >= averageHeight * 1.25:
        playerAverageHeightJudge = 'VeryHeight'
    elif playerHeight < averageHeight * 1.45 and playerHeight >= averageHeight * 1.35:
        playerAverageHeightJudge = 'SuperHeight'
    elif playerHeight > averageHeight * 1.45:
        playerAverageHeightJudge = 'ExtremelyHeight'
    elif playerHeight < averageHeight * 0.95 and playerHeight >= averageHeight * 0.85:
        playerAverageHeightJudge = 'SlilghtlyLow'
    elif playerHeight < averageHeight * 0.85 and playerHeight >= averageHeight * 0.75:
        playerAverageHeightJudge = 'Low'
    elif playerHeight < averageHeight * 0.75 and playerHeight >= averageHeight * 0.65:
        playerAverageHeightJudge = 'VeryLow'
    elif playerHeight < averageHeight * 0.65 and playerHeight >= averageHeight * 0.55:
        playerAverageHeightJudge = 'SuperLow'
    elif playerHeight < averageHeight * 0.55:
        playerAverageHeightJudge = 'ExtremelyLow'
    targetJudge = 'Target'
    if characterId == 0:
        targetJudge = 'Self'
    return {
        "SelfSex":playerSex,
        "TargetSex":characterData.Sex,
        "AgeJudge":ageJudge,
        "AgeTem":ageTem,
        "SelfAgeTem":playerAgeTem,
        "SelfAge":playerAge,
        "TargetAge":targetAge,
        "AverageHeight":heightJudge,
        "AverageStature":bodyFatJudge,
        "RelativeHeight":playerHeightJudge,
        "RelativeStature":playerBodyFatJudge,
        "PlayerAverageHeight":playerAverageHeightJudge,
        "PlayerAverageStature":playerAverageBodyFatJudge,
        "Target":targetJudge
    }

def getStatureText(characterId:str) -> list:
    '''
    按角色Id获取身材描述信息
    Keyword arguments:
    characterId -- 角色Id
    '''
    statureJudge = judgeCharacterStature(characterId)
    for priority in CacheContorl.statureDescritionPrioritionData:
        for descript in CacheContorl.statureDescritionPrioritionData[priority]:
            if judgeStatureDescription(statureJudge,TextLoading.getGameData(TextLoading.statureDescriptionPath)['Priority'][priority][descript]['Condition']):
                return TextLoading.getGameData(TextLoading.statureDescriptionPath)['Priority'][priority][descript]['Description']
    return ''

def judgeStatureDescription(statureJudge,descriptionData):
    '''
    角色的身材信息
    Keyword arguments:
    statureJudge -- 身材校验信息
    descriptionData -- 身材描述数据
    '''
    for judge in descriptionData:
        if judge == 'SelfAge' and statureJudge['SelfAge'] < descriptionData[judge]:
            return False
        elif judge == 'TargetAge' and statureJudge['TargetAge'] < descriptionData[judge]:
            return False
        else:
            if descriptionData[judge] != statureJudge[judge]:
                return False
    return True

def getCharacterAbbreviationsInfo(characterId:int) -> str:
    '''
    按角色id获取角色缩略信息文本
    Keyword arguments:
    characterId -- 角色id
    '''
    characterData = CacheContorl.characterData['character'][characterId]
    characterIdInfo = TextLoading.getTextData(TextLoading.stageWordPath, '0')
    characterIdText = f"{characterIdInfo}{characterId}"
    characterName = characterData.Name
    characterSex = characterData.Sex
    characterSexInfo = TextLoading.getTextData(TextLoading.stageWordPath, '2')
    characterSexTextData = TextLoading.getTextData(TextLoading.rolePath,'Sex')
    characterSexText = characterSexTextData[characterSex]
    characterSexText = characterSexInfo + characterSexText
    characterAge = characterData.Age
    characterAgeInfo = TextLoading.getTextData(TextLoading.stageWordPath, '3')
    characterAgeText = characterAgeInfo + str(characterAge)
    characterHpAndMpText = AttrPrint.getHpAndMpText(characterId)
    characterIntimate = characterData.Intimate
    characterIntimateInfo = TextLoading.getTextData(TextLoading.stageWordPath, '16')
    characterIntimateText = characterIntimateInfo + str(characterIntimate)
    characterGraces = characterData.Graces
    characterGracesInfo = TextLoading.getTextData(TextLoading.stageWordPath, '17')
    characterGracesText = characterGracesInfo + str(characterGraces)
    abbreviationsInfo = characterIdText + ' ' + characterName + ' ' + characterSexText + ' ' + characterAgeText + ' ' + characterHpAndMpText + ' ' + characterIntimateText + ' ' + characterGracesText
    return abbreviationsInfo

def getCharacterDormitoryPathText(characterId:str) -> str:
    '''
    获取角色宿舍路径描述信息
    Keyword arguments:
    characterId -- 角色Id
    Return arguments:
    mapPathStr -- 宿舍路径描述文本
    '''
    dormitory = CacheContorl.characterData['character'][characterId].Dormitory
    dormitoryPath = MapHandle.getMapSystemPathForStr(dormitory)
    mapList = MapHandle.getMapHierarchyListForScenePath(dormitoryPath,[])
    mapPathText = TextLoading.getTextData(TextLoading.stageWordPath,'143')
    mapList.reverse()
    for nowMap in mapList:
        nowMapMapSystemStr = MapHandle.getMapSystemPathStrForList(nowMap)
        mapName = CacheContorl.mapData[nowMapMapSystemStr]['MapName']
        mapPathText += mapName + '-'
    mapPathText += CacheContorl.sceneData[dormitory]['SceneName']
    return mapPathText
