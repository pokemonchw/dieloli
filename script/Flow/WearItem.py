from script.Core import CacheContorl,GameConfig
from script.Panel import WearItemPanel

def sceneSeeCharacterWearItem(characterId:str):
    '''
    在场景中查看角色穿戴道具列表的流程
    Keyword arguments:
    characterId -- 角色Id
    '''
    while 1:
        inputS = WearItemPanel.seeCharacterWearItemPanelForPlayer(characterId)

def getCharacterWearItemPageMax(characterId:str):
    '''
    计算角色可穿戴道具列表页数
    Keyword arguments:
    characterId -- 角色Id
    '''
    wearItemMax = len(CacheContorl.characterData['character'][characterId]['WearItem']['Item'])
    pageIndex - GameConfig.see_character_wearitem_max
    if wearItemMax - pageIndex < 0:
        return 0
    elif wearItemMax % pageIndex == 0:
        return wearItemMax / pageIndex - 1
    else:
        return int(wearItemMax / pageIndex)
