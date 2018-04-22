import os
import core.data as data
import core.TextLoading as textload
import script.ProportionalBar as proportionalbar
import core.CacheContorl as cache
from core.GameConfig import language
from core.pycfg import gamepath

roleAttrPath = os.path.join(gamepath,'data',language,'RoleAttributes.json')
roleAttrData = data._loadjson(roleAttrPath)
sexData = roleAttrData['Sex']
equipmentPath = os.path.join(gamepath,'data',language,'Equipment.json')
equipmentData = data._loadjson(equipmentPath)

#获取性经验文本
def getSexExperienceText(sexList,sexName):
    mouthExperience = textload.getTextData(textload.stageWordId,'19') + str(sexList['mouthExperience'])
    bosomExperience = textload.getTextData(textload.stageWordId,'20') + str(sexList['bosomExperience'])
    vaginaExperience = textload.getTextData(textload.stageWordId,'21') + str(sexList['vaginaExperience'])
    clitorisExperience = textload.getTextData(textload.stageWordId,'22') + str(sexList['clitorisExperience'])
    anusExperience = textload.getTextData(textload.stageWordId,'23') + str(sexList['anusExperience'])
    penisExperience = textload.getTextData(textload.stageWordId,'24') + str(sexList['penisExperience'])
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
    mouthText = textload.getTextData(textload.stageWordId,'25') + getGradeTextColor(sexGradeList['mouthGrade'])
    bosomText = textload.getTextData(textload.stageWordId,'26') + getGradeTextColor(sexGradeList['bosomGrade'])
    vaginaText = textload.getTextData(textload.stageWordId,'27') + getGradeTextColor(sexGradeList['vaginaGrade'])
    clitorisText = textload.getTextData(textload.stageWordId,'28') + getGradeTextColor(sexGradeList['clitorisGrade'])
    anusText = textload.getTextData(textload.stageWordId,'29') + getGradeTextColor(sexGradeList['anusGrade'])
    penisText = textload.getTextData(textload.stageWordId,'30') + getGradeTextColor(sexGradeList['penisGrade'])
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
    painLevelFix = textload.getTextData(textload.stageWordId,'31')
    happyLevelFix = textload.getTextData(textload.stageWordId,'32')
    yieldLevelFix = textload.getTextData(textload.stageWordId,'33')
    fearLevelFix = textload.getTextData(textload.stageWordId,'34')
    resistanceLevelFix = textload.getTextData(textload.stageWordId,'35')
    LVText = textload.getTextData(textload.stageWordId,'36')
    levelList = [painLevel,happyLevel,yieldLevel,fearLevel,resistanceLevel]
    levelFixList = [painLevelFix,happyLevelFix,yieldLevelFix,fearLevelFix,resistanceLevelFix]
    levelTextList = []
    levelBarList = []
    for i in range(0,len(levelList)):
        levelTextList.append(levelFixList[i] + LVText + levelList[i])
    for i in range(0,len(levelList)):
        levelBarList.append(proportionalbar.getCountBar(levelTextList[i],3,levelList[i],'engravingemptybar'))
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
    coatText = textload.getTextData(textload.stageWordId,"41") + coatText
    pantsText = textload.getTextData(textload.stageWordId, "42") + pantsText
    shoesText = textload.getTextData(textload.stageWordId, "43") + shoesText
    socksText = textload.getTextData(textload.stageWordId, "44") + socksText
    underwearText = textload.getTextData(textload.stageWordId, "45") + underwearText
    braText = textload.getTextData(textload.stageWordId, "46") + braText
    underpantsText = textload.getTextData(textload.stageWordId, "47") + underpantsText
    leggingsText = textload.getTextData(textload.stageWordId, "48") + leggingsText
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
    headText = textload.getTextData(textload.stageWordId,"49") + headText
    eyeText = textload.getTextData(textload.stageWordId,"50") + eyeText
    earText = textload.getTextData(textload.stageWordId,"51") + earText
    mouthText = textload.getTextData(textload.stageWordId,"52") + mouthText
    finesseText = textload.getTextData(textload.stageWordId,"53") + finesseText
    fingerText = textload.getTextData(textload.stageWordId,"54") + fingerText
    chestText = textload.getTextData(textload.stageWordId,"55") + chestText
    privatesText = textload.getTextData(textload.stageWordId,"56") + privatesText
    anusText = textload.getTextData(textload.stageWordId,"57") + anusText
    otherText = textload.getTextData(textload.stageWordId,"58") + otherText
    sexItemTextList = [
        headText,eyeText,earText,mouthText,finesseText,
        fingerText,chestText,privatesText,anusText,otherText
    ]
    return sexItemTextList

# 获取金钱信息文本
def getGoldText(playerId):
    goldData = cache.playObject['object'][playerId]['Gold']
    goldData = str(goldData)
    moneyText = textload.getTextData(textload.stageWordId,'66')
    goldText = textload.getTextData(textload.stageWordId,'67')
    goldText = goldText + goldData + moneyText
    return goldText