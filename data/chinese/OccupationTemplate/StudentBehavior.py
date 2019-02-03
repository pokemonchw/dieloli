import os
from script.Core import CacheContorl,GameData,GamePathConfig,GameConfig
from script.Design import GameTime

gamePath = GamePathConfig.gamepath
language = GameConfig.language

# npc状态判断
def objectStateJudge(self,objectId):
    objectData = CacheContorl.playObject['object'][objectId]
    objectState = objectData['State']
    objectName = objectData['Name']
    characterData = CacheContorl.npcTemData[int(objectId) - 1]
    objectBehavior = 'object' + objectState + 'Behavior'
    behavior = getattr(self,objectBehavior)
    behavior(objectId,objectData,characterData)

#休闲状态行为
def arderBehavior(self,objectId, objectData, characterData):
    nowWeekDay = GameTime.getWeekDate()
    if nowWeekDay in range(1,5):
        nowTimeSlice = GameTime.getNowTimeSlice()
    else:
        pass
