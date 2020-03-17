from script.Core import CacheContorl
from script.Design import GameTime
from script.Behavior import Default

# 职业行为总控制
def behaviorInit(characterId):
    characterState = CacheContorl.characterData['character'][characterId].State
    if characterState in behaviorList:
        behaviorList[characterState](characterId)
    else:
        Default.behaviorList[characterState](characterId)

#休闲状态行为
def arderBehavior(characterId):
    nowWeekDay = GameTime.getWeekDate()
    if nowWeekDay in range(1,5):
        nowTimeSlice = GameTime.getNowTimeSlice()

behaviorList = {
    "arder":arderBehavior
}

