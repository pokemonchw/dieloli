from script.Core import GameConfig,GameInit,PyCmd,CacheContorl
from script.Design import CharacterHandle
from script.Panel import SeePlayerListPanel

# 用于查看角色列表的流程
def seePlayerList_func(oldPanel):
    maxPage = getPlayerListPageMax()
    inputS = []
    seePlayerListPanelInput = SeePlayerListPanel.seePlayerListPanel(maxPage)
    startId = len(seePlayerListPanelInput)
    inputS = inputS + seePlayerListPanelInput
    askForSeePlayerListPanelInput = SeePlayerListPanel.askForSeePlayerListPanel(startId)
    inputS = inputS + askForSeePlayerListPanelInput
    yrn = GameInit.askfor_All(inputS)
    yrn = str(yrn)
    playerIdList = CharacterHandle.getCharacterIdList()
    PyCmd.clr_cmd()
    pageId = int(CacheContorl.panelState['SeePlayerListPanel'])
    if yrn in playerIdList:
        import script.Flow.SeePlayerAttr as seeplayerattr
        CacheContorl.playObject['objectId'] = yrn
        seeplayerattr.seeAttrOnEveryTime_func('SeePlayerListPanel',oldPanel)
    elif yrn == str(startId):
        if pageId == 0:
            CacheContorl.panelState['SeePlayerListPanel'] = str(maxPage)
            seePlayerList_func(oldPanel)
        else:
            pageId = str(pageId - 1)
            CacheContorl.panelState['SeePlayerListPanel'] = pageId
            seePlayerList_func(oldPanel)
    elif yrn == str(startId + 1):
        if oldPanel == 'MainFramePanel':
            import script.Flow.Main as mainframe
            CacheContorl.playObject['objectId'] = '0'
            mainframe.mainFrame_func()
        else:
            pass
    elif yrn == str(startId + 2):
        if pageId == maxPage:
            CacheContorl.panelState['SeePlayerListPanel'] = '0'
            seePlayerList_func(oldPanel)
        else:
            pageId = str(pageId + 1)
            CacheContorl.panelState['SeePlayerListPanel'] = pageId
            seePlayerList_func(oldPanel)
    pass

# 角色列表页计算
def getPlayerListPageMax():
    playerMax = CharacterHandle.getCharacterIndexMax()
    playerPageShow = int(GameConfig.playerlist_show)
    if playerMax - playerPageShow < 0:
        return 0
    elif playerMax % playerPageShow == 0:
        return playerMax / playerPageShow
    else:
        return playerMax / playerPageShow + 1
    pass