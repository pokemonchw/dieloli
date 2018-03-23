import core.CacheContorl as cache
import core.ValueHandle as valuehandle

# 获取角色最大数量
def getCharacterIndexMax():
    playerData = cache.playObject['object']
    playerMax = valuehandle.indexDictKeysMax(playerData) - 1
    return playerMax

# 获取角色id列表
def getCharacterIdList():
    playerData = cache.playObject['object']
    playerList = valuehandle.dictKeysToList(playerData)
    return playerList