import core.TextLoading as textload
import core.EraPrint as eprint
import core.CacheContorl as cache
import core.GameConfig as config
import script.CharacterHandle as characterhandle
import script.AttrPrint as attrprint
import script.Ans as ans
import core.PyCmd as pycmd
import core.TextHandle as text

# 查看角色列表面板
def seePlayerListPanel(maxPage):
    titleText = textload.getTextData(textload.stageWordId,'74')
    eprint.plt(titleText)
    inputS = []
    pageId = int(cache.panelState['SeePlayerListPanel'])
    pageShow = int(config.playerlist_show)
    maxPage = int(maxPage)
    playerMax = characterhandle.getCharacterIndexMax()
    if pageId == maxPage:
        showPageStart = pageShow * (pageId)
        showPageOver = showPageStart + (playerMax - showPageStart)
    else:
        showPageOver = pageShow * pageId
        showPageStart = showPageOver - pageShow
    for i in range(showPageStart,showPageOver + 1):
        playerId = str(i)
        cmdId = i - showPageStart
        cmdIdText = ans.idIndex(cmdId)
        playerData = cache.playObject['object'][playerId]
        playerIdInfo = textload.getTextData(textload.stageWordId,'0')
        playerIdText = playerIdInfo + playerId
        playerName = playerData['Name']
        playerSex = playerData['Sex']
        playerSexInfo = textload.getTextData(textload.stageWordId,'2')
        playerSexText = playerSexInfo + playerSex
        playerAge = playerData['Age']
        playerAgeInfo = textload.getTextData(textload.stageWordId,'3')
        playerAgeText = playerAgeInfo + str(playerAge)
        playerHpAndMpText = attrprint.getHpAndMpText(playerId)
        playerIntimate = playerData['Intimate']
        playerIntimateInfo = textload.getTextData(textload.stageWordId, '16')
        playerIntimateText = playerIntimateInfo + playerIntimate
        playerGraces = playerData['Graces']
        playerGracesInfo = textload.getTextData(textload.stageWordId, '17')
        playerGracesText = playerGracesInfo + playerGraces
        cmdText = playerIdText + ' ' + playerName + ' ' + playerSexText +' ' + playerAgeText +' ' + playerHpAndMpText + ' ' + playerIntimateText + ' ' + playerGracesText
        cmdIdTextIndex = text.getTextIndex(cmdIdText)
        windowWidth = int(config.text_width)
        textWidth = windowWidth - cmdIdTextIndex
        cmdText = text.align(cmdText,'center',textWidth=textWidth)
        cmdText = cmdIdText + ' ' + cmdText
        cmdId = str(cmdId)
        eprint.plittleline()
        pycmd.pcmd(cmdText, cmdId, None)
        inputS.append(cmdId)
        eprint.p('\n')
    pageText = '(' + str(pageId) + '/' + str(maxPage) + ')'
    eprint.printPageLine(sample = '-',string = pageText)
    eprint.p('\n')
    return inputS

# 询问切换角色列表页面面板
def askForSeePlayerListPanel(startId):
    yrn = ans.optionint(ans.seeplayerlist,3,'left',askfor=False,cmdSize='center',startId=startId)
    return yrn