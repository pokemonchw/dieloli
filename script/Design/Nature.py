import random
from script.Core import TextLoading,CacheContorl

def initCharacterNature(characterId:str):
    '''
    初始化角色性格
    Keyword arguments:
    characterId -- 角色Id
    '''
    natureList = TextLoading.getGameData(TextLoading.naturePath)
    natureData = {aDimension:{bDimension:random.uniform(0,100) for bDimension in natureList[aDimension]} for aDimension in natureList}
    CacheContorl.characterData['character'][characterId]['Nature'] = natureData
