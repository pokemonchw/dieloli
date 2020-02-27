from script.Core import CacheContorl,TextLoading,EraPrint
from script.Design import AttrText

def seeCharacterSocialContactPanel(characterId:str):
    '''
    查看角色社交信息面板
    Keyword arguments:
    characterId -- 角色Id
    '''
    socialContactTextData = TextLoading.getTextData(TextLoading.stageWordPath,'144')
    characterSocialContact = CacheContorl.characterData['character'][characterId].SocialContact
    for social in socialContactTextData:
        EraPrint.sontitleprint(socialContactTextData[social])
        if characterSocialContact[social] == {}:
            EraPrint.p(TextLoading.getTextData(TextLoading.messagePath,'40'))
        else:
            size = 10
            if len(characterSocialContact[social]) < 10:
                size = len(characterSocialContact[social])
            nameList = [CacheContorl.characterData['character'][characterId].Name for characterId in characterSocialContact[social]]
            EraPrint.plist(nameList,size,'center')
