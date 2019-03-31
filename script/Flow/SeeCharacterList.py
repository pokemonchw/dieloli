from script.Core import GameConfig,GameInit,PyCmd,CacheContorl
from script.Design import CharacterHandle
from script.Panel import SeeCharacterListPanel

characterPageShow = int(GameConfig.characterlist_show)

# 用于查看角色列表的流程
def seeCharacterList_func(oldPanel):
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
            seeCharacterList_func(oldPanel)
        else:
            pageId = str(pageId - 1)
            CacheContorl.panelState['SeeCharacterListPanel'] = pageId
            seeCharacterList_func(oldPanel)
    elif yrn == str(startId + 1):
        if oldPanel == 'MainFramePanel':
            import script.Flow.Main as mainframe
            CacheContorl.characterData['characterId'] = '0'
            CacheContorl.panelState['SeeCharacterListPanel'] = '0'
            mainframe.mainFrame_func()
        else:
            pass
    elif yrn == str(startId + 2):
        if pageId == maxPage:
            CacheContorl.panelState['SeeCharacterListPanel'] = '0'
            seeCharacterList_func(oldPanel)
        else:
            pageId = str(pageId + 1)
            CacheContorl.panelState['SeeCharacterListPanel'] = pageId
            seeCharacterList_func(oldPanel)
    elif yrn in characterIdList:
        yrn = str(int(yrn) + characterPageShow * pageId)
        CacheContorl.characterData['characterId'] = yrn
        SeeCharacterListPanel.seeAttrOnEveryTime_func('SeeCharacterListPanel',oldPanel)

# 角色列表页计算
def getCharacterListPageMax():
    characterMax = CharacterHandle.getCharacterIndexMax()
    if characterMax - characterPageShow < 0:
        return 0
    elif characterMax % characterPageShow == 0:
        return characterMax / characterPageShow - 1
    else:
        return int(characterMax / characterPageShow)
