import os
import core.data as data
import script.TextLoading as textload
import script.ProportionalBar as proportionalbar
from core.GameConfig import language
from core.pycfg import gamepath

roleAttrPath = os.path.join(gamepath,'data',language,'RoleAttributes.json')
roleAttrData = data._loadjson(roleAttrPath)
sexData = roleAttrData['Sex']

#获取性经验文本
def getSexExperienceText(sexList,sexName):
    mouthExperience = textload.loadStageWordText('19') + str(sexList['mouthExperience'])
    bosomExperience = textload.loadStageWordText('20') + str(sexList['bosomExperience'])
    vaginaExperience = textload.loadStageWordText('21') + str(sexList['vaginaExperience'])
    clitorisExperience = textload.loadStageWordText('22') + str(sexList['clitorisExperience'])
    anusExperience = textload.loadStageWordText('23') + str(sexList['anusExperience'])
    penisExperience = textload.loadStageWordText('24') + str(sexList['penisExperience'])
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
    mouthText = textload.loadStageWordText('25') + getGradeTextColor(sexGradeList['mouthGrade'])
    bosomText = textload.loadStageWordText('26') + getGradeTextColor(sexGradeList['bosomGrade'])
    vaginaText = textload.loadStageWordText('27') + getGradeTextColor(sexGradeList['vaginaGrade'])
    clitorisText = textload.loadStageWordText('28') + getGradeTextColor(sexGradeList['clitorisGrade'])
    anusText = textload.loadStageWordText('29') + getGradeTextColor(sexGradeList['anusGrade'])
    penisText = textload.loadStageWordText('30') + getGradeTextColor(sexGradeList['penisGrade'])
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
    if sexGrade == 'G':
        sexGrade = '<levelg>' + sexGrade + '</levelg>'
    elif sexGrade == 'F':
        sexGrade = '<levelf>' + sexGrade + '</levelf>'
    elif sexGrade == 'E':
        sexGrade = '<levele>' + sexGrade + '</levele>'
    elif sexGrade == 'D':
        sexGrade = '<leveld>' + sexGrade + '</leveld>'
    elif sexGrade == 'C':
        sexGrade = '<levelc>' + sexGrade + '</levelc>'
    elif sexGrade == 'B':
        sexGrade = '<levelb>' + sexGrade + '</levelb>'
    elif sexGrade == 'A':
        sexGrade = '<levela>' + sexGrade + '</levela>'
    elif sexGrade == 'EX':
        sexGrade = '<levelex>' + sexGrade + '</levelex>'
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
    painLevelFix = textload.loadStageWordText('31')
    happyLevelFix = textload.loadStageWordText('32')
    yieldLevelFix = textload.loadStageWordText('33')
    fearLevelFix = textload.loadStageWordText('34')
    resistanceLevelFix = textload.loadStageWordText('35')
    LVText = textload.loadStageWordText('36')
    levelList = [painLevel,happyLevel,yieldLevel,fearLevel,resistanceLevel]
    levelFixList = [painLevelFix,happyLevelFix,yieldLevelFix,fearLevelFix,resistanceLevelFix]
    levelTextList = []
    levelBarList = []
    for i in range(0,len(levelList)):
        levelTextList.append(levelFixList[i] + LVText + levelList[i])
    for i in range(0,len(levelList)):
        levelBarList.append(proportionalbar.getCountBar(levelTextList[i],3,levelList[i],'engravingfull','✡','✡','engravingempty'))
    return levelBarList
