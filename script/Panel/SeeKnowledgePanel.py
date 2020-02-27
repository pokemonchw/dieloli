from script.Core import CacheContorl,TextLoading,EraPrint
from script.Design import AttrText

def seeCharacterKnowledgePanel(characterId:str):
    '''
    查看角色技能信息面板
    Keyword arguments:
    characterId -- 角色Id
    '''
    knowledgeTextData = TextLoading.getGameData(TextLoading.knowledge)
    characterKnowledge = CacheContorl.characterData['character'][characterId].Knowledge
    for knowledge in knowledgeTextData:
        EraPrint.sontitleprint(knowledgeTextData[knowledge]['Name'])
        if knowledge in characterKnowledge:
            infoList = [knowledgeTextData[knowledge]['Knowledge'][skill]['Name'] + ":" + AttrText.getLevelColorText(characterKnowledge[knowledge][skill]) for skill in characterKnowledge[knowledge]]
            EraPrint.plist(infoList,6,'center')
