import core.CacheContorl as cache

# 获取角色属性
def getAttrData(playerId):
    playerId = str(playerId)
    attrData = cache.playObject['object'][playerId]
    return attrData

# 设置角色属性
def setAttrData(playerId,attrId,attr):
    playerId = str(playerId)
    cache.playObject['object'][playerId][attrId] = attr
    pass