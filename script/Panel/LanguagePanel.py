from script.Core import CacheContorl,TextLoading,EraPrint
from script.Design import AttrText

def seeCharacterLanguagePanel(characterId:str):
    '''
    查看角色语言能力面板
    Keyword arguments:
    characterId -- 角色Id
    '''
    languageTextData = TextLoading.getGameData(TextLoading.languageSkillsPath)
    characterLanguage = CacheContorl.characterData['character'][characterId]['Language']
    infoList = [languageTextData[language]['Name'] + ":" + AttrText.getLevelColorText(characterLanguage[language]) for language in characterLanguage]
    EraPrint.plist(infoList,4,'center')
