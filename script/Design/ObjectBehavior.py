import os,importlib
from script.Core import CacheContorl,GamePathConfig,GameConfig,GameData
from script.Design import CharacterHandle

gamePath = GamePathConfig.gamepath
language = GameConfig.language
characterListPath = os.path.join(gamePath,'data',language,'character')

# npc行为总控制
def initObjectBehavior():
    npcList = CharacterHandle.getCharacterIdList()
    npcList.remove('0')
    for npc in npcList:
        objectStateJudge(npc)

# npc状态判断
def objectStateJudge(objectId):
    objectData = CacheContorl.playObject['object'][objectId]
    objectState = objectData['State']
    objectName = objectData['Name']
    objectFile = os.path.join(characterListPath,objectName,'AttrTemplate.json')
    objectFileData = GameData._loadjson(objectFile)
    if objectState == 'Arder':
        objectArderBehavior(objectId,objectData,objectFileData)
    else:
        pass

# 休闲状态行为
def objectArderBehavior(objectId,objectData,objectFileData):
    if objectFileData['AdvNpc'] == '1':
        objectOccupationJudge(objectId,objectData,objectFileData)
    else:
        pass

# npc职业判断
def objectOccupationJudge(objectId, objectData,objectFileData):
    try:
        objectOccupation = objectFileData['Occupation']
        templateFile = 'data.' + language + '.OccupationTemplate.' + objectOccupation + 'Behavior'
        template = importlib.import_module(templateFile)
        template.arderBehavior(objectId, objectData, objectFileData)
    except:
        objectAgeJudge(objectId, objectData, objectFileData)

# Npc年龄判断
def objectAgeJudge(objectId,objectData,objectFileData):
    objectAge = objectData['Age']
    objectAge = int(objectAge)
    if objectAge <= 18:
        studentArderBehavior(objectId, objectData,objectFileData)
    else:
        teacherArderBehavior(objectId, objectData, objectFileData)

# 学生休闲行为
def studentArderBehavior(objectId,objectData,objectFileData):
    templateFile = 'data.' + language + '.OccupationTemplate.' + 'Student' + 'Behavior'
    template = importlib.import_module(templateFile)
    template.arderBehavior(objectId, objectData, objectFileData)

# 教师休闲行为
def teacherArderBehavior(objectId,objectData,objectFileData):
    templateFile = 'data.' + language + '.OccupationTemplate.' + 'Teacher' + 'Behavior'
    template = importlib.import_module(templateFile)
    template.arderBehavior(objectId, objectData, objectFileData)