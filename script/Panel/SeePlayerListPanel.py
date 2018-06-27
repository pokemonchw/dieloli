from script.Core import TextLoading,EraPrint,CacheContorl,GameConfig,PyCmd,TextHandle
from script.Design import CharacterHandle,CmdButtonQueue,AttrText

# 查看角色列表面板
def seePlayerListPanel(maxPage):
    titleText = TextLoading.getTextData(TextLoading.stageWordId,'74')
    EraPrint.plt(titleText)
    inputS = []
    pageId = int(CacheContorl.panelState['SeePlayerListPanel'])
    pageShow = int(GameConfig.playerlist_show)
    maxPage = int(maxPage)
    playerMax = CharacterHandle.getCharacterIndexMax()
    if pageId == maxPage:
        showPageStart = pageShow * (pageId)
        showPageOver = showPageStart + (playerMax - showPageStart)
    else:
        showPageOver = pageShow * pageId
        showPageStart = showPageOver - pageShow
    for i in range(showPageStart,showPageOver + 1):
        playerId = str(i)
        cmdId = i - showPageStart
        cmdIdText = CmdButtonQueue.idIndex(cmdId)
        cmdText = AttrText.getPlayerAbbreviationsInfo(playerId)
        cmdIdTextIndex = TextHandle.getTextIndex(cmdIdText)
        windowWidth = int(GameConfig.text_width)
        textWidth = windowWidth - cmdIdTextIndex
        cmdText = TextHandle.align(cmdText,'center',textWidth=textWidth)
        cmdText = cmdIdText + ' ' + cmdText
        cmdId = str(cmdId)
        EraPrint.plittleline()
        PyCmd.pcmd(cmdText, cmdId, None)
        inputS.append(cmdId)
        EraPrint.p('\n')
    pageText = '(' + str(pageId) + '/' + str(maxPage) + ')'
    EraPrint.printPageLine(sample = '-',string = pageText)
    EraPrint.p('\n')
    return inputS

# 询问切换角色列表页面面板
def askForSeePlayerListPanel(startId):
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.seeplayerlist, 3, 'left', askfor=False, cmdSize='center', startId=startId)
    return yrn