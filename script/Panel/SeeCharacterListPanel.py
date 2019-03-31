from script.Core import TextLoading,EraPrint,CacheContorl,GameConfig,PyCmd,TextHandle
from script.Design import CharacterHandle,CmdButtonQueue,AttrText

# 查看角色列表面板
def seeCharacterListPanel(maxPage):
    titleText = TextLoading.getTextData(TextLoading.stageWordPath,'74')
    EraPrint.plt(titleText)
    inputS = []
    pageId = int(CacheContorl.panelState['SeeCharacterListPanel'])
    pageShow = int(GameConfig.characterlist_show)
    maxPage = int(maxPage)
    characterMax = CharacterHandle.getCharacterIndexMax()
    if pageId == maxPage:
        showPageStart = pageShow * pageId
        showPageOver = showPageStart + (characterMax + 1 - showPageStart)
    else:
        showPageOver = pageShow * (pageId + 1)
        showPageStart = showPageOver - pageShow
    for i in range(showPageStart,showPageOver):
        characterId = str(i)
        cmdId = i - showPageStart
        cmdIdText = CmdButtonQueue.idIndex(cmdId)
        cmdText = AttrText.getCharacterAbbreviationsInfo(characterId)
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
def askForSeeCharacterListPanel(startId):
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.seecharacterlist, 3, 'left', askfor=False, cmdSize='center', startId=startId)
    return yrn
