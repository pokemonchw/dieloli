from script.Core import CacheContorl

# 获取角色属性
def getAttrData(playerId):
    playerId = str(playerId)
    attrData = CacheContorl.playObject['object'][playerId]
    return attrData

# 设置角色属性
def setAttrData(playerId,attrId,attr):
    playerId = str(playerId)
    CacheContorl.playObject['object'][playerId][attrId] = attr
    pass