import os
from script.Core import TextLoading,GameData,CacheContorl,GameConfig,GamePathConfig
from script.Design import ProportionalBar,AttrPrint

language = GameConfig.language
gamepath = GamePathConfig.gamepath

roleAttrPath = os.path.join(gamepath,'data',language,'RoleAttributes.json')
roleAttrData = GameData._loadjson(roleAttrPath)
sexData = roleAttrData['Sex']
equipmentPath = os.path.join(gamepath,'data',language,'Equipment.json')
equipmentData = GameData._loadjson(equipmentPath)

#获取性经验文本
def getSexExperienceText(sexList,sexName):
    mouthExperience = TextLoading.getTextData(TextLoading.stageWordId,'19') + str(sexList['mouthExperience'])
    bosomExperience = TextLoading.getTextData(TextLoading.stageWordId,'20') + str(sexList['bosomExperience'])
    vaginaExperience = TextLoading.getTextData(TextLoading.stageWordId,'21') + str(sexList['vaginaExperience'])
    clitorisExperience = TextLoading.getTextData(TextLoading.stageWordId,'22') + str(sexList['clitorisExperience'])
    anusExperience = TextLoading.getTextData(TextLoading.stageWordId,'23') + str(sexList['anusExperience'])
    penisExperience = TextLoading.getTextData(TextLoading.stageWordId,'24') + str(sexList['penisExperience'])
    sexExperienceText = []
    if sexName == sexData[0]:
        sexExperienceText = [mouthExperience,bosomExperience,anusExperience,penisExperience]
    elif sexName == sexData[1]:
        sexExperienceText = [mouthExperience,bosomExperience,vaginaExperience,clitorisExperience,anusExperience]
    elif sexName == sexData[2]:
        sexExperienceText = [mouthExperience,bosomExperience,vaginaExperience,clitorisExperience,anusExperience,penisExperience]
    elif sexName == sexData[3]:
        sexExperienceText = [mouthExperience,bosomExperience,anusExperience]
    return sexExperienceText

#获取性等级文本
def getSexGradeTextList(sexGradeList,sexName):
    mouthText = TextLoading.getTextData(TextLoading.stageWordId,'25') + getGradeTextColor(sexGradeList['mouthGrade'])
    bosomText = TextLoading.getTextData(TextLoading.stageWordId,'26') + getGradeTextColor(sexGradeList['bosomGrade'])
    vaginaText = TextLoading.getTextData(TextLoading.stageWordId,'27') + getGradeTextColor(sexGradeList['vaginaGrade'])
    clitorisText = TextLoading.getTextData(TextLoading.stageWordId,'28') + getGradeTextColor(sexGradeList['clitorisGrade'])
    anusText = TextLoading.getTextData(TextLoading.stageWordId,'29') + getGradeTextColor(sexGradeList['anusGrade'])
    penisText = TextLoading.getTextData(TextLoading.stageWordId,'30') + getGradeTextColor(sexGradeList['penisGrade'])
    sexGradeTextList = []
    if sexName == sexData[0]:
        sexGradeTextList = [mouthText,bosomText,anusText,penisText]
    elif sexName == sexData[1]:
        sexGradeTextList = [mouthText,bosomText,vaginaText,clitorisText,anusText]
    elif sexName == sexData[2]:
        sexGradeTextList = [mouthText,bosomText,vaginaText,clitorisText,anusText,penisText]
    elif sexName == sexData[3]:
        sexGradeTextList = [mouthText,bosomText,anusText]
    return sexGradeTextList

# 处理等级富文本
def getGradeTextColor(sexGrade):
    lowerGrade = sexGrade.lower()
    sexGrade = '<level' + lowerGrade + '>' + sexGrade + '</level' + lowerGrade + '>'
    return sexGrade

# 获取特征文本
def getFeaturesStr(fList):
    featuresListStr = ''
    featuresListText = ['Age',"Chastity",'Disposition','Courage','SelfConfidence','Friends','Figure',
                        'Sex','AnimalInternal','AnimalExternal','Charm'
                        ]
    for i in range(0,len(featuresListText)):
        try:
            featureText = fList[featuresListText[i]]
            if featureText != '':
                featuresListStr = featuresListStr + '[' + featureText + ']'
            else:
                pass
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
    painLevelFix = TextLoading.getTextData(TextLoading.stageWordId,'31')
    happyLevelFix = TextLoading.getTextData(TextLoading.stageWordId,'32')
    yieldLevelFix = TextLoading.getTextData(TextLoading.stageWordId,'33')
    fearLevelFix = TextLoading.getTextData(TextLoading.stageWordId,'34')
    resistanceLevelFix = TextLoading.getTextData(TextLoading.stageWordId,'35')
    LVText = TextLoading.getTextData(TextLoading.stageWordId,'36')
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
    coatText = TextLoading.getTextData(TextLoading.stageWordId,"41") + coatText
    pantsText = TextLoading.getTextData(TextLoading.stageWordId, "42") + pantsText
    shoesText = TextLoading.getTextData(TextLoading.stageWordId, "43") + shoesText
    socksText = TextLoading.getTextData(TextLoading.stageWordId, "44") + socksText
    underwearText = TextLoading.getTextData(TextLoading.stageWordId, "45") + underwearText
    braText = TextLoading.getTextData(TextLoading.stageWordId, "46") + braText
    underpantsText = TextLoading.getTextData(TextLoading.stageWordId, "47") + underpantsText
    leggingsText = TextLoading.getTextData(TextLoading.stageWordId, "48") + leggingsText
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
    headText = TextLoading.getTextData(TextLoading.stageWordId,"49") + headText
    eyeText = TextLoading.getTextData(TextLoading.stageWordId,"50") + eyeText
    earText = TextLoading.getTextData(TextLoading.stageWordId,"51") + earText
    mouthText = TextLoading.getTextData(TextLoading.stageWordId,"52") + mouthText
    finesseText = TextLoading.getTextData(TextLoading.stageWordId,"53") + finesseText
    fingerText = TextLoading.getTextData(TextLoading.stageWordId,"54") + fingerText
    chestText = TextLoading.getTextData(TextLoading.stageWordId,"55") + chestText
    privatesText = TextLoading.getTextData(TextLoading.stageWordId,"56") + privatesText
    anusText = TextLoading.getTextData(TextLoading.stageWordId,"57") + anusText
    otherText = TextLoading.getTextData(TextLoading.stageWordId,"58") + otherText
    sexItemTextList = [
        headText,eyeText,earText,mouthText,finesseText,
        fingerText,chestText,privatesText,anusText,otherText
    ]
    return sexItemTextList

# 获取金钱信息文本
def getGoldText(playerId):
    goldData = CacheContorl.playObject['object'][playerId]['Gold']
    goldData = str(goldData)
    moneyText = TextLoading.getTextData(TextLoading.stageWordId,'66')
    goldText = TextLoading.getTextData(TextLoading.stageWordId,'67')
    goldText = goldText + goldData + moneyText
    return goldText


# 获取角色缩略信息
def getPlayerAbbreviationsInfo(playerId):
    playerData = CacheContorl.playObject['object'][playerId]
    playerIdInfo = TextLoading.getTextData(TextLoading.stageWordId, '0')
    playerIdText = playerIdInfo + playerId
    playerName = playerData['Name']
    playerSex = playerData['Sex']
    playerSexInfo = TextLoading.getTextData(TextLoading.stageWordId, '2')
    playerSexText = playerSexInfo + playerSex
    playerAge = playerData['Age']
    playerAgeInfo = TextLoading.getTextData(TextLoading.stageWordId, '3')
    playerAgeText = playerAgeInfo + str(playerAge)
    playerHpAndMpText = AttrPrint.getHpAndMpText(playerId)
    playerIntimate = playerData['Intimate']
    playerIntimateInfo = TextLoading.getTextData(TextLoading.stageWordId, '16')
    playerIntimateText = playerIntimateInfo + playerIntimate
    playerGraces = playerData['Graces']
    playerGracesInfo = TextLoading.getTextData(TextLoading.stageWordId, '17')
    playerGracesText = playerGracesInfo + playerGraces
    abbreviationsInfo = playerIdText + ' ' + playerName + ' ' + playerSexText + ' ' + playerAgeText + ' ' + playerHpAndMpText + ' ' + playerIntimateText + ' ' + playerGracesText
    return abbreviationsInfo