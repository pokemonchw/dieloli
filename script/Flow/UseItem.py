from script.Core import CacheContorl

def sceneSeeCharacterItem(characterId:str):
    '''
    在场景中查看角色道具列表的流程
    Keyword arguments:
    characterId -- 角色Id
    '''
    while 1:
        pass

def openCharacterBag():
    '''
    打开主角背包查看道具列表流程
    '''
    while 1:
        pass

def getCharacterItemPageMax(characterId:str):
    '''
    计算角色道具列表页数
    Keyword arguments:
    characterId -- 角色Id
    '''
    itemMax = len(CacheContorl.characterData['character'][characterId].Item)
    pageIndex - GameConfig.see_character_item_max
    if itemMax - pageIndex < 0:
        return 0
    elif itemMax % pageIndex == 0:
        return itemMax / pageIndex - 1
    else:
        return int(itemMax / pageIndex)
