from script.Core import CacheContorl

# 获取角色属性
def getAttrData(characterId):
    characterId = str(characterId)
    return CacheContorl.characterData['character'][characterId]

# 设置角色属性
def setAttrData(characterId,attrId,attr):
    characterId = str(characterId)
    CacheContorl.characterData['character'][characterId][attrId] = attr
