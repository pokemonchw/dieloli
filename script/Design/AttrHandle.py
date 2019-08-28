from script.Core import CacheContorl

def getAttrData(characterId:str) -> dict:
    '''
    获取指定id的角色数据
    Keyword arguments:
    characterId -- 角色id
    '''
    return CacheContorl.characterData['character'][characterId]

def setAttrData(characterId:str,attrId:str,attr:dict):
    '''
    设置指定角色的特定属性数据
    Keyword arguments:
    characterId -- 角色id
    attrId -- 属性id
    attr -- 属性数据
    '''
    characterId = str(characterId)
    CacheContorl.characterData['character'][characterId][attrId] = attr
