import os,importlib
from script.Core import CacheContorl,GamePathConfig,GameConfig,GameData
from script.Design import CharacterHandle

gamePath = GamePathConfig.gamepath
language = GameConfig.language
characterListPath = os.path.join(gamePath,'data',language,'character')

class ObjectBehaviorClass(object):

    # npc行为总控制
    def initObjectBehavior(self):
        npcList = CharacterHandle.getCharacterIdList()
        npcList.remove('0')
        for npc in npcList:
            self.objectOccupationJudge(npc)

    # npc职业判断
    def objectOccupationJudge(self,objectId):
        objectData = CacheContorl.playObject['object'][objectId]
        objectName = objectData['Name']
        objectFile = os.path.join(characterListPath, objectName, 'AttrTemplate.json')
        objectFileData = GameData._loadjson(objectFile)
        try:
            objectOccupation = objectFileData['Occupation']
            templateFile = 'data.' + language + '.OccupationTemplate.' + objectOccupation + 'Behavior'
        except:
            objectAge = int(objectData['Age'])
            if objectAge <= 18:
                objectOccupation = "Student"
            else:
                objectOccupation = "Teacher"
            templateFile = 'data.' + language + '.OccupationTemplate.' + objectOccupation + 'Behavior'
        template = importlib.import_module(templateFile)
        template.arderBehavior(objectId, objectData, objectFileData)
