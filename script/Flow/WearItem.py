from script.Core import CacheContorl,GameConfig,FlowHandle,PyCmd
from script.Panel import WearItemPanel

def sceneSeeCharacterWearItem(characterId:str):
    '''
    在场景中查看角色穿戴道具列表的流程
    Keyword arguments:
    characterId -- 角色Id
    '''
    while 1:
        WearItemPanel.seeCharacterWearItemPanelForPlayer(characterId)

def wearCharacterItem():
    '''
    查看并更换角色穿戴道具流程
    '''
    characterId = CacheContorl.characterData['characterId']
    while 1:
        inputs = WearItemPanel.seeCharacterWearItemPanelForPlayer(characterId)
        startId = len(inputs)
        inputs += WearItemPanel.seeCharacterWearItemCmdPanel(startId)
        yrn = FlowHandle.askfor_All(inputs)
        PyCmd.clr_cmd()
        if yrn == str(startId):
            CacheContorl.nowFlowId == 'main'
        else:
            wearItemInfoTextData = TextLoading.getTextData(TextLoading.stageWordPath,'49')
            changeWearItem(list(wearItemInfoTextData.keys())[int(yrn)])

def changeWearItem(itemType:str):
    '''
    更换角色穿戴道具流程
    Keyword arguments:
    itemType -- 道具类型
    '''
    characterId = CacheContorl.characterData['characterId']
    itemData = CacheContorl.characterData['character'][characterId]
    maxPage = getCharacterWearItemPageMax(characterId)
    inputS = WearItemPanel.seeCharacterWearItemListPanel(characterId,itemType,maxPage)

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