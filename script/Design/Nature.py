import random
from script.Core import TextLoading,CacheContorl

def initCharacterNature(characterId:str):
    '''
    初始化角色性格
    Keyword arguments:
    characterId -- 角色Id
    '''
    natureList = TextLoading.getGameData(TextLoading.naturePath)
    natureData = {bDimension:random.uniform(0,100) for aDimension in natureList for bDimension in natureList[aDimension]['Factor']}
    CacheContorl.characterData['character'][characterId]['Nature'] = natureData
