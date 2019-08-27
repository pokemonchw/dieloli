from script.Core import GameConfig,GameInit,CacheContorl
from script.Design import CharacterHandle
from script.Panel import SeeCharacterListPanel

characterPageShow = int(GameConfig.characterlist_show)

def seeCharacterList_func():
    '''
    用于查看角色列表的流程
    '''
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

def getCharacterListPageMax():
    '''
    计算角色列表总页数，公式为角色总数/每页显示角色数
    '''
    characterMax = CharacterHandle.getCharacterIndexMax()
    if characterMax - characterPageShow < 0:
        return 0
    elif characterMax % characterPageShow == 0:
        return characterMax / characterPageShow - 1
    else:
        return int(characterMax / characterPageShow)
