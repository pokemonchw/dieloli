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
boysNameMax = len(boysNameListData)
girlsNameMax = len(girlsNameListData)
sortFamilyIndex = sorted(familyNameListData.items(),key=lambda x:x[1])

familyRegionList = ValueHandle.getReginList(familyNameListData)
boysRegionList = ValueHandle.getReginList(boysNameListData)
girlsRegionList = ValueHandle.getReginList(girlsNameListData)

familyRegionIntList = list(map(int,familyRegionList))
boysRegionIntList = list(map(int,boysRegionList))
girlsRegionIntList = list(map(int,girlsRegionList))

#获取性经验文本
def getSexExperienceText(sexList,sexName):
    mouthExperience = TextLoading.getTextData(TextLoading.stageWordPath,'19') + str(sexList['mouthExperience'])
    bosomExperience = TextLoading.getTextData(TextLoading.stageWordPath,'20') + str(sexList['bosomExperience'])
    vaginaExperience = TextLoading.getTextData(TextLoading.stageWordPath,'21') + str(sexList['vaginaExperience'])
    clitorisExperience = TextLoading.getTextData(TextLoading.stageWordPath,'22') + str(sexList['clitorisExperience'])
    anusExperience = TextLoading.getTextData(TextLoading.stageWordPath,'23') + str(sexList['anusExperience'])
    penisExperience = TextLoading.getTextData(TextLoading.stageWordPath,'24') + str(sexList['penisExperience'])
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

#获取性等级文本
def getSexGradeTextList(sexGradeList,sexName):
    mouthText = TextLoading.getTextData(TextLoading.stageWordPath,'25') + getGradeTextColor(sexGradeList['mouthGrade'])
    bosomText = TextLoading.getTextData(TextLoading.stageWordPath,'26') + getGradeTextColor(sexGradeList['bosomGrade'])
    vaginaText = TextLoading.getTextData(TextLoading.stageWordPath,'27') + getGradeTextColor(sexGradeList['vaginaGrade'])
    clitorisText = TextLoading.getTextData(TextLoading.stageWordPath,'28') + getGradeTextColor(sexGradeList['clitorisGrade'])
    anusText = TextLoading.getTextData(TextLoading.stageWordPath,'29') + getGradeTextColor(sexGradeList['anusGrade'])
    penisText = TextLoading.getTextData(TextLoading.stageWordPath,'30') + getGradeTextColor(sexGradeList['penisGrade'])
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
def getGradeTextColor(sexGrade):
    lowerGrade = sexGrade.lower()
    sexGrade = '<level' + lowerGrade + '>' + sexGrade + '</level' + lowerGrade + '>'
    return sexGrade

familyIndexMax = familyRegionIntList[len(familyRegionIntList) - 1]
# 按性别随机生成姓名
def getRandomNameForSex(sexGrade):
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
        nameRandom = random.randint(1,girlsNameMax)
        nameRegionIndex = bisect.bisect_left(girlsRegionIntList,nameRandom)
        nameRegion = girlsRegionIntList[nameRegionIndex]
        name = girlsRegionList[str(nameRegion)]
    else:
        nameRandom = random.randint(1,boysNameMax)
        nameRegionIndex = bisect.bisect_left(boysRegionIntList,nameRandom)
        nameRegion = boysRegionIntList[nameRegionIndex]
        name = boysRegionList[str(nameRegion)]
    return familyName + name

# 获取性别对应文本
def getSexText(sexId):
    data = roleAttrData['Sex']
    sexText = data[sexId]
    return sexText

# 获取特征文本
def getFeaturesStr(fList):
    featuresListStr = ''
    featuresListText = [
        'Age',"Chastity",'Disposition','Courage','SelfConfidence','Friends','Figure',
        'Sex','AnimalInternal','AnimalExternal','Charm'
    ]
    for i in range(0,len(featuresListText)):
        try:
            featureText = fList[featuresListText[i]]
            if featureText != '':
                featuresListStr = featuresListStr + '[' + featureText + ']'
        except KeyError:
            pass
    return featuresListStr

# 获取刻印文本
def getEngravingText(eList):
    painLevel = eList["Pain"]
    happyLevel = eList["Happy"]
    yieldLevel = eList["Yield"]
    fearLevel = eList["Fear"]
    resistanceLevel = eList["Resistance"]
    painLevelFix = TextLoading.getTextData(TextLoading.stageWordPath,'31')
    happyLevelFix = TextLoading.getTextData(TextLoading.stageWordPath,'32')
    yieldLevelFix = TextLoading.getTextData(TextLoading.stageWordPath,'33')
    fearLevelFix = TextLoading.getTextData(TextLoading.stageWordPath,'34')
    resistanceLevelFix = TextLoading.getTextData(TextLoading.stageWordPath,'35')
    LVText = TextLoading.getTextData(TextLoading.stageWordPath,'36')
    levelList = [painLevel,happyLevel,yieldLevel,fearLevel,resistanceLevel]
    levelFixList = [painLevelFix,happyLevelFix,yieldLevelFix,fearLevelFix,resistanceLevelFix]
    levelTextList = []
    levelBarList = []
    for i in range(0,len(levelList)):
        levelTextList.append(levelFixList[i] + LVText + levelList[i])
    for i in range(0,len(levelList)):
        levelBarList.append(ProportionalBar.getCountBar(levelTextList[i], 3, levelList[i], 'engravingemptybar'))
    return levelBarList

# 获取服装列表文本
def getClothingText(clothingList):
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

# 获取性道具文本
def getSexItemText(sexItemList):
    headid = sexItemList["Head"]
    eyeid = sexItemList["Eye"]
    earid = sexItemList["Ear"]
    mouthid = sexItemList["Mouth"]
    finesseid = sexItemList["Finesse"]
    fingerid = sexItemList["Finger"]
    chestid = sexItemList["Chest"]
    privatesid = sexItemList["Privates"]
    anusid = sexItemList["Anus"]
    otherid = sexItemList["Other"]
    sexItemData = equipmentData["SexItem"]
    headText = sexItemData["Head"][headid]
    eyeText = sexItemData["Eye"][eyeid]
    earText = sexItemData["Ear"][earid]
    mouthText = sexItemData["Mouth"][mouthid]
    finesseText = sexItemData["Finesse"][finesseid]
    fingerText = sexItemData["Finger"][fingerid]
    chestText = sexItemData["Chest"][chestid]
    privatesText = sexItemData["Privates"][privatesid]
    anusText = sexItemData["Anus"][anusid]
    otherText = sexItemData["Other"][otherid]
    headText = TextLoading.getTextData(TextLoading.stageWordPath,"49") + headText
    eyeText = TextLoading.getTextData(TextLoading.stageWordPath,"50") + eyeText
    earText = TextLoading.getTextData(TextLoading.stageWordPath,"51") + earText
    mouthText = TextLoading.getTextData(TextLoading.stageWordPath,"52") + mouthText
    finesseText = TextLoading.getTextData(TextLoading.stageWordPath,"53") + finesseText
    fingerText = TextLoading.getTextData(TextLoading.stageWordPath,"54") + fingerText
    chestText = TextLoading.getTextData(TextLoading.stageWordPath,"55") + chestText
    privatesText = TextLoading.getTextData(TextLoading.stageWordPath,"56") + privatesText
    anusText = TextLoading.getTextData(TextLoading.stageWordPath,"57") + anusText
    otherText = TextLoading.getTextData(TextLoading.stageWordPath,"58") + otherText
    sexItemTextList = [
        headText,eyeText,earText,mouthText,finesseText,
        fingerText,chestText,privatesText,anusText,otherText
    ]
    return sexItemTextList

# 获取金钱信息文本
def getGoldText(characterId):
    goldData = CacheContorl.characterData['character'][characterId]['Gold']
    goldData = str(goldData)
    moneyText = TextLoading.getTextData(TextLoading.stageWordPath,'66')
    goldText = TextLoading.getTextData(TextLoading.stageWordPath,'67')
    goldText = goldText + goldData + moneyText
    return goldText


# 获取角色缩略信息
def getCharacterAbbreviationsInfo(characterId):
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
