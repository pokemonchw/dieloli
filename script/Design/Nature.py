import random
from script.Core import TextLoading,CacheContorl

def getRandomNature():
    '''
    初始化角色性格
    '''
    natureList = TextLoading.getGameData(TextLoading.naturePath)
    natureData = {bDimension:random.uniform(0,100) for aDimension in natureList for bDimension in natureList[aDimension]['Factor']}
    return natureData
