from script.Core import TextLoading,CacheContorl,EraPrint,GameConfig
from script.Design import AttrText,CmdButtonQueue

def seeCharacterWearItemPanelForPlayer(characterId:int) -> list:
    '''
    用于场景中查看穿戴道具列表的控制面板
    Keyword arguments:
    characterId -- 角色Id
    changeButton -- 将角色穿戴道具列表绘制成按钮的开关
    '''
    EraPrint.plt(TextLoading.getTextData(TextLoading.stageWordPath, '40'))
    EraPrint.p(AttrText.getSeeAttrPanelHeadCharacterInfo(characterId))
    EraPrint.pline('.')
    if characterId == 0:
        return seeCharacterWearItemPanel(characterId,True)
    else:
        return seeCharacterWearItemPanel(characterId,False)

def seeCharacterWearItemPanel(characterId:int,changeButton:bool) -> list:
    '''
    用于查看角色穿戴道具列表的面板
    Keyword arguments:
    characterId -- 角色Id
    changeButton -- 将角色穿戴道具列表绘制成按钮的开关
    '''
    wearItemInfoTextData = TextLoading.getTextData(TextLoading.stageWordPath,'49')
    wearData = CacheContorl.characterData['character'][characterId].WearItem['Wear']
    wearItemTextData = {}
    itemData = CacheContorl.characterData['character'][characterId].WearItem['Item']
    wearItemButtonList = []
    inputS = []
    for wearType in wearData:
        wearId = wearData[wearType]
        if wearId == '':
            wearItemButtonList.append(wearItemInfoTextData[wearType] + ':' + TextLoading.getTextData(TextLoading.stageWordPath,'117'))
        else:
            wearItemButtonList.append(wearItemInfoTextData[wearType] + ':' + itemData[wearType][wearId]['Name'])
            wearItemTextData[wearType] = itemData[wearType][wearId]['Name']
    if changeButton:
        inputS = [str(i) for i in range(len(wearData))]
        CmdButtonQueue.optionint(None,4,'left',True,False,'center',0,wearItemButtonList,)
    else:
        EraPrint.plist(wearItemButtonList,4,'center')
    return inputS

def seeCharacterWearItemListPanel(characterId:int,itemType:str,maxPage:int) -> list:
    '''
    用于查看角色可穿戴道具列表的面板
    Keyword arguments:
    characterId -- 用户Id
    itemType -- 道具类型
    maxPage -- 道具列表最大页数
    '''
    EraPrint.pl()
    characterWearItemData = CacheContorl.characterData['character'][characterId].WearItem['Item'][itemType]
    nowPageId = int(CacheContorl.panelState["SeeCharacterWearItemListPanel"])
    nowPageMax = GameConfig.see_character_wearitem_max
    nowPageStartId = nowPageId * nowPageMax
    nowPageEndId = nowPageStartId + nowPageMax
    if characterWearItemData == {}:
        EraPrint.pl(TextLoading.getTextData(TextLoading.messagePath,'38'))
        return []
    if nowPageEndId > len(characterWearItemData.keys()):
        nowPageEndId = len(characterWearItemData.keys())

def seeCharacterWearItemCmdPanel(startId:int) -> list:
    '''
    查看角色已穿戴道具列表的控制面板
    Keyword arguments:
    startId -- 命令起始Id
    '''
    EraPrint.pline()
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.seecharacterwearclothes,cmdSize='center',askfor=False,startId=startId)
    return yrn
