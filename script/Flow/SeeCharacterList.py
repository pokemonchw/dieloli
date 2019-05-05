from script.Core import GameConfig,GameInit,PyCmd,CacheContorl
from script.Design import CharacterHandle
from script.Panel import SeeCharacterListPanel

characterPageShow = int(GameConfig.characterlist_show)

# 用于查看角色列表的流程
def seeCharacterList_func():
    while(True):
        maxPage = getCharacterListPageMax()
        inputS = []
        seeCharacterListPanelInput = SeeCharacterListPanel.seeCharacterListPanel(maxPage)
        startId = len(seeCharacterListPanelInput)
        inputS = inputS + seeCharacterListPanelInput
        askForSeeCharacterListPanelInput = SeeCharacterListPanel.askForSeeCharacterListPanel(startId)
        inputS = inputS + askForSeeCharacterListPanelInput
        yrn = GameInit.askfor_All(inputS)
        yrn = str(yrn)
        characterIdList = CharacterHandle.getCharacterIdList()
        pageId = int(CacheContorl.panelState['SeeCharacterListPanel'])
        if yrn == str(startId):
            if pageId == 0:
                CacheContorl.panelState['SeeCharacterListPanel'] = str(maxPage)
            else:
                pageId = str(pageId - 1)
                CacheContorl.panelState['SeeCharacterListPanel'] = pageId
        elif yrn == str(startId + 1):
            CacheContorl.characterData['characterId'] = '0'
            CacheContorl.panelState['SeeCharacterListPanel'] = '0'
            CacheContorl.nowFlowId = CacheContorl.oldFlowId
            break
        elif yrn == str(startId + 2):
            if pageId == maxPage:
                CacheContorl.panelState['SeeCharacterListPanel'] = '0'
            else:
                pageId = str(pageId + 1)
                CacheContorl.panelState['SeeCharacterListPanel'] = pageId
        elif yrn in characterIdList:
            yrn = str(int(yrn) + characterPageShow * pageId)
            CacheContorl.characterData['characterId'] = yrn
            CacheContorl.nowFlowId = 'see_character_attr'
            CacheContorl.tooOldFlowId = CacheContorl.oldFlowId
            CacheContorl.oldFlowId = 'see_character_list'
            break

# 角色列表页计算
def getCharacterListPageMax():
    characterMax = CharacterHandle.getCharacterIndexMax()
    if characterMax - characterPageShow < 0:
        return 0
    elif characterMax % characterPageShow == 0:
        return characterMax / characterPageShow - 1
    else:
        return int(characterMax / characterPageShow)
