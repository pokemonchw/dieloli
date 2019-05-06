from script.Core import CacheContorl
from script.Design import GameTime

#休闲状态行为
def arderBehavior(characterId):
    nowWeekDay = GameTime.getWeekDate()
    if nowWeekDay in range(1,5):
        nowTimeSlice = GameTime.getNowTimeSlice()
    else:
        pass
