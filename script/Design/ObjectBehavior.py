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
            self.objectStateJudge(npc)

    # npc状态判断
    def objectStateJudge(self,objectId):
        objectData = CacheContorl.playObject['object'][objectId]
        objectState = objectData['State']
        objectName = objectData['Name']
        objectFile = os.path.join(characterListPath, objectName, 'AttrTemplate.json')
        objectFileData = GameData._loadjson(objectFile)
        objectBehavior = 'object' + objectState + 'Behavior'
        behavior = getattr(self,objectBehavior)
        behavior(objectId,objectData,objectFileData)

    # 休闲状态行为
    def objectArderBehavior(self,objectId, objectData, objectFileData):
        if objectFileData['AdvNpc'] == '1':
            self.objectOccupationJudge(objectId, objectData, objectFileData)
        else:
            pass

    # npc职业判断
    def objectOccupationJudge(self,objectId, objectData, objectFileData):
        try:
            objectOccupation = objectFileData['Occupation']
            templateFile = 'data.' + language + '.OccupationTemplate.' + objectOccupation + 'Behavior'
            template = importlib.import_module(templateFile)
            template.arderBehavior(objectId, objectData, objectFileData)
        except:
            self.objectAgeJudge(objectId, objectData, objectFileData)

    # Npc年龄判断
    def objectAgeJudge(self,objectId, objectData, objectFileData):
        objectAge = objectData['Age']
        objectAge = int(objectAge)
        if objectAge <= 18:
            self.studentArderBehavior(objectId, objectData, objectFileData)
        else:
            self.teacherArderBehavior(objectId, objectData, objectFileData)

    # 学生休闲行为
    def studentArderBehavior(self,objectId, objectData, objectFileData):
        templateFile = 'data.' + language + '.OccupationTemplate.' + 'Student' + 'Behavior'
        template = importlib.import_module(templateFile)
        template.arderBehavior(objectId, objectData, objectFileData)

    # 教师休闲行为
    def teacherArderBehavior(self,objectId, objectData, objectFileData):
        templateFile = 'data.' + language + '.OccupationTemplate.' + 'Teacher' + 'Behavior'
        template = importlib.import_module(templateFile)
        template.arderBehavior(objectId, objectData, objectFileData)