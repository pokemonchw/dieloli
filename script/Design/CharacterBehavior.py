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

# npc行为总控制
def initCharacterBehavior():
    npcList = CharacterHandle.getCharacterIdList()
    npcList.remove('0')
    for npc in npcList:
        characterOccupationJudge(npc)

# npc职业判断
def characterOccupationJudge(characterId):
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
