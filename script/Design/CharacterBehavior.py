import os,importlib
from script.Core import CacheContorl,GamePathConfig,GameConfig
from script.Design import CharacterHandle

gamePath = GamePathConfig.gamepath
language = GameConfig.language
characterListPath = os.path.join(gamePath,'data',language,'character')

class CharacterBehaviorClass(object):

    # npc行为总控制
    def initCharacterBehavior(self):
        npcList = CharacterHandle.getCharacterIdList()
        npcList.remove('0')
        for npc in npcList:
            self.characterOccupationJudge(npc)

    # npc职业判断
    def characterOccupationJudge(self,characterId):
        characterData = CacheContorl.characterData['character'][characterId]
        characterTemData = CacheContorl.npcTemData[int(characterId) - 1]
        try:
            characterOccupation = characterTemData['Occupation']
            templateFile = 'data.' + language + '.OccupationTemplate.' + characterOccupation + 'Behavior'
        except:
            characterAge = int(characterData['Age'])
            if characterAge <= 18:
                characterOccupation = "Student"
            else:
                characterOccupation = "Teacher"
            templateFile = 'data.' + language + '.OccupationTemplate.' + characterOccupation + 'Behavior'
        template = importlib.import_module(templateFile)
        template.arderBehavior(self,characterId, characterData, characterTemData)
