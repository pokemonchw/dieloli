from script.Core import TextLoading,CacheContorl,CmdButtonQueue,PyCmd,EraPrint
from script.Design import AttrText

def seeCharacterItemPanel(characterId:str) -> list:
    '''
    用于查看角色背包道具列表的面板
    Keyword arguments:
    characterId -- 角色Id
    '''
    EraPrint.plt(TextLoading.getTextData(TextLoading.stageWordPath, '40'))
    EraPrint.p(AttrText.getSeeAttrPanelHeadCharacterInfo(characterId))
    EraPrint.pline('.')
    characterItemData = CacheContorl.characterData['character'][characterId]['Item']
    if len(characterItemData) == 0:
        EraPrint.pl(TextLoading.getTextData(TextLoading.messagePath,'36'))
        return []
    nowPageId = int(CacheContorl.panelState['SeeCharacterItemListPanel'])
    nowPageMax = GameConfig.see_character_item_max
    nowPageStartId = nowPageId * nowPageMax
    nowPageEndId = nowPageStartId + nowPageMax
    if nowPageEndId > len(characterItemData.keys()):
        nowPageEndId = len(characterItemData.keys())
    inputS = []
    index = 0
    for i in range(nowPageStartId,nowPageEndId):
        itemId = list(characterItemData.keys())[i]
        itemData = characterItemData[itemId]
        itemText = itemData['ItemName'] + ' ' + TextLoading.getTextData(TextLoading.stageWordPath,'136') + str(itemData['ItemNum'])
        if characterId == '0':
            idInfo = CmdButtonQueue.idIndex(index)
            cmdText = idInfo + drawText
            PyCmd.pcmd(cmdText,index,None)
        else:
            EraPrint.p(drawText)
        index += 1
        inputS.append(str(index))
        EraPrint.p('\n')
    return inputS

def seeCharacterItemInfoPanel(characterId:str,itemId:str):
    '''
    用于查看角色道具信息的面板
    Keyword arguments:
    characterId -- 角色Id
    itemId -- 道具Id
    '''
    titleText = TextLoading.getTextData(TextLoading.stageWordPath,'38')
    EraPrint.plt(titleText)
    EraPrint.p(AttrText.getSeeAttrPanelHeadCharacterInfo(characterId))
    EraPrint.pline('.')
    itemData = CacheContorl.characterData['character'][characterId]['Item'][itemId]
    EraPrint.pl(TextLoading.getTextData(TextLoading.stageWordPath,128) + itemData['ItemName'])
    EraPrint.pl(TextLoading.getTextData(TextLoading.stageWordPath,'131') + itemData['ItemInfo'])
