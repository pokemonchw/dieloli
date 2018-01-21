import os
import core.data as data
import script.TextLoading as textload
from core.GameConfig import language
from core.pycfg import gamepath

roleAttrPath = os.path.join(gamepath,'data',language,'RoleAttributes.json')
roleAttrData = data._loadjson(roleAttrPath)
sexData = roleAttrData['Sex']

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

def getSexGradeTextList(sexGradeList,sexName):
    mouthText = textload.loadStageWordText('25') + sexGradeList['mouthGrade']
    bosomText = textload.loadStageWordText('26') + sexGradeList['bosomGrade']
    vaginaText = textload.loadStageWordText('27') + sexGradeList['vaginaGrade']
    clitorisText = textload.loadStageWordText('28') + sexGradeList['clitorisGrade']
    anusText = textload.loadStageWordText('29') + sexGradeList['anusGrade']
    penisText = textload.loadStageWordText('30') + sexGradeList['penisGrade']
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

def getFeaturesStr(fList):
    featuresListStr = ''
    featuresListText = ['Age',"Chastity",'Disposition','SelfConfidence','Friends','Figure',
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