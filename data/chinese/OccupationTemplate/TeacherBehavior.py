import os,importlib
from script.Core import CacheContorl,GameData,GamePathConfig,GameConfig

gamePath = GamePathConfig.gamepath
language = GameConfig.language
characterListPath = os.path.join(gamePath,'data',language,'character')

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

#休闲状态行为
def arderBehavior(self,objectId, objectData, objectFileData):
    if objectFileData['AdvNpc'] == '1':
        pass
    else:
        pass
