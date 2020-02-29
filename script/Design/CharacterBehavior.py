import os,importlib
from script.Core import CacheContorl,GamePathConfig,GameConfig
from script.Design import CharacterHandle
from script.Behavior import Student,Teacher

gamePath = GamePathConfig.gamepath
language = GameConfig.language
characterListPath = os.path.join(gamePath,'data',language,'character')

behaviorTemData = {
    "Student":Student,
    "Teacher":Teacher
}

def initCharacterBehavior():
    '''
    角色行为树总控制
    '''
    for npc in CacheContorl.characterData['character']:
        if npc == 0:
            continue
        characterOccupationJudge(npc)

def characterOccupationJudge(characterId:int):
    '''
    判断角色职业并指定对应行为树
    Keyword arguments:
    characterId -- 角色id
    '''
    characterData = CacheContorl.characterData['character'][characterId]
    characterTemData = CacheContorl.npcTemData[characterId - 1]
    if 'Occupation' in characterTemData and characterTemData['Occupation'] in behaviorTemData:
        characterOccupation = characterTemData['Occupation']
    else:
        characterAge = int(characterData.Age)
        if characterAge <= 18:
            characterOccupation = "Student"
        else:
            characterOccupation = "Teacher"
    template = behaviorTemData[characterOccupation]
    template.behaviorInit(characterId)
