import os
from Core import CacheContorl,GamePathConfig,GameConfig,GameData
from Design import CharacterHandle,GameTime

gamePath = GamePathConfig.gamepath
language = GameConfig.language
characterListPath = os.path.join(gamePath,'data',language,'character')

# npc行为总控制
def initObjectBehavior():
    npcList = CharacterHandle.getCharacterIdList()
    for npc in npcList:
        objectStateJudge(npc)

# npc状态判断
def objectStateJudge(objectId):
    objectData = CacheContorl.playObject['object'][objectId]
    objectState = objectData['State']
    if objectState == 'Arder':
        objectArderBehavior(objectId,objectData)
    else:
        pass

# 休闲状态行为
def objectArderBehavior(objectId,objectData):
    objectName = objectData['Name']
    objectFile = os.path.join(characterListPath,objectName,'AttrTemplate.json')
    objectFileData = GameData._loadjson(objectFile)
    if objectFileData['AdvNpc'] == '1':
        objectAgeJudge(objectId,objectData,objectFileData)
    else:
        pass

# Npc年龄判断
def objectAgeJudge(objectId,objectData,objectFileData):
    objectAge = objectData['Age']
    objectAge = int(objectAge)
    if objectAge <= 18:
        studentArderBehavior(objectId, objectData,objectFileData)
    else:
        pass

# 学生休闲行为
def studentArderBehavior(objectId,objectData,objectFileData):
    pass