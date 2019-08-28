import os,importlib
from script.Core import CacheContorl,GamePathConfig,GameConfig
from script.Design import CharacterHandle

gamePath = GamePathConfig.gamepath
language = GameConfig.language
characterListPath = os.path.join(gamePath,'data',language,'character')

behaviorTemTextData = {
    "Student":'script.Behavior.Student',
    "Teacher":'script.Behavior.Teacher'
}

behaviorTemData = {
    "Student":importlib.import_module(behaviorTemTextData['Student']),
    "Teacher":importlib.import_module(behaviorTemTextData['Teacher'])
}

def initCharacterBehavior():
    '''
    角色行为树总控制
    '''
    npcList = CharacterHandle.getCharacterIdList()
    npcList.remove('0')
    for npc in npcList:
        characterOccupationJudge(npc)

def characterOccupationJudge(characterId:str):
    '''
    判断角色职业并指定对应行为树
    Keyword arguments:
    characterId -- 角色id
    '''
    characterData = CacheContorl.characterData['character'][characterId]
    characterTemData = CacheContorl.npcTemData[int(characterId) - 1]
    if 'Occupation' in characterTemData and characterTemData['Occupation'] in behaviorTemData:
        characterOccupation = characterTemData['Occupation']
    else:
        characterAge = int(characterData['Age'])
        if characterAge <= 18:
            characterOccupation = "Student"
        else:
            characterOccupation = "Teacher"
    template = behaviorTemData[characterOccupation]
    template.behaviorInit(characterId)
