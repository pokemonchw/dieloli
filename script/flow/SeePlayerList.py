import script.Panel.SeePlayerListPanel as seeplayerlistpanel
import script.CharacterHandle as characterhandle
import core.GameConfig as config
import core.game as game
import core.PyCmd as pycmd
import core.CacheContorl as cache

# 用于查看角色列表的流程
def seePlayerList_func(oldPanel):
    maxPage = getPlayerListPageMax()
    inputS = []
    seePlayerListPanelInput = seeplayerlistpanel.seePlayerListPanel(maxPage)
    startId = len(seePlayerListPanelInput)
    inputS = inputS + seePlayerListPanelInput
    askForSeePlayerListPanelInput = seeplayerlistpanel.askForSeePlayerListPanel(startId)
    inputS = inputS + askForSeePlayerListPanelInput
    yrn = game.askfor_All(inputS)
    yrn = str(yrn)
    playerIdList = characterhandle.getCharacterIdList()
    pycmd.clr_cmd()
    pageId = int(cache.panelState['SeePlayerListPanel'])
    if yrn in playerIdList:
        import script.flow.SeePlayerAttr as seeplayerattr
        cache.playObject['objectId'] = yrn
        seeplayerattr.seeAttrOnEveryTime_func('SeePlayerListPanel',oldPanel)
    elif yrn == str(startId):
        if pageId == 0:
            cache.panelState['SeePlayerListPanel'] = str(maxPage)
            seePlayerList_func(oldPanel)
        else:
            pageId = str(pageId - 1)
            cache.panelState['SeePlayerListPanel'] = pageId
            seePlayerList_func(oldPanel)
    elif yrn == str(startId + 1):
        if oldPanel == 'MainFramePanel':
            import script.flow.MainFrame as mainframe
            mainframe.mainFrame_func()
        else:
            pass
    elif yrn == str(startId + 2):
        if pageId == maxPage:
            cache.panelState['SeePlayerListPanel'] = '0'
            seePlayerList_func(oldPanel)
        else:
            pageId = str(pageId + 1)
            cache.panelState['SeePlayerListPanel'] = pageId
            seePlayerList_func(oldPanel)
    pass

# 角色列表页计算
def getPlayerListPageMax():
    playerMax = characterhandle.getCharacterIndexMax()
    playerPageShow = int(config.playerlist_show)
    if playerMax - playerPageShow < 0:
        return 0
    elif playerMax % playerPageShow == 0:
        return playerMax / playerPageShow
    else:
        return playerMax / playerPageShow + 1
    pass