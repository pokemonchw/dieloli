from script.Core import CacheContorl

def getAttrData(characterId):
    '''
    获取指定id的角色数据
    Keyword arguments:
    characterId -- 角色id
    '''
    characterId = str(characterId)
    return CacheContorl.characterData['character'][characterId]

def setAttrData(characterId,attrId,attr):
    '''
    设置指定角色的特定属性数据
    Keyword arguments:
    characterId -- 角色id
    attrId -- 属性id
    attr -- 属性数据
    '''
    characterId = str(characterId)
    CacheContorl.characterData['character'][characterId][attrId] = attr
