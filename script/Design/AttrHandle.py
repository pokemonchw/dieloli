from script.Core import CacheContorl

# 获取角色属性
def getAttrData(characterId):
    characterId = str(characterId)
    attrData = CacheContorl.characterData['character'][characterId]
    return attrData

# 设置角色属性
def setAttrData(characterId,attrId,attr):
    characterId = str(characterId)
    CacheContorl.characterData['character'][characterId][attrId] = attr
