from script.Core import CacheContorl,TextLoading,EraPrint
from script.Design import AttrText,AttrHandle

def seeCharacterSexExperiencePanel(characterId:str):
    '''
    查看角色性经验面板
    Keyword arguments:
    characterId -- 角色Id
    '''
    EraPrint.plittleline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.stageWordPath, '5'))
    characterData = AttrHandle.getAttrData(characterId)
    characterSexGradeList = characterData['SexGrade']
    characterSex = CacheContorl.characterData['character'][characterId]['Sex']
    characterSexGradeTextList = AttrText.getSexGradeTextList(characterSexGradeList, characterSex)
    EraPrint.plist(characterSexGradeTextList, 4, 'center')
    EraPrint.pl(TextLoading.getTextData(TextLoading.stageWordPath, '7'))
    characterEngraving = characterData['Engraving']
    characterEngravingText = AttrText.getEngravingText(characterEngraving)
    EraPrint.plist(characterEngravingText, 3, 'center')
