import core.EraPrint as eprint
import core.TextLoading as textload
import script.GameTime as gametime
import core.CacheContorl as cache
import script.AttrText as attrtext
import script.AttrPrint as attrprint
import script.AttrHandle as attrhandle
import core.PyCmd as pycmd
import script.Ans as ans

# 游戏主页流程
def mainFramePanel():
    cmdList = []
    playerId = cache.playObject['objectId']
    playerData = attrhandle.getAttrData(playerId)
    titleText = textload.getTextData(textload.stageWordId, '64')
    eprint.plt(titleText)
    dateText = gametime.getDateText()
    eprint.p(dateText)
    eprint.p(' ')
    weekDateText = gametime.getWeekDayText()
    eprint.p(weekDateText)
    eprint.p(' ')
    playerName = playerData['Name']
    pycmd.pcmd(playerName,playerName,None)
    cmdList.append(playerName)
    eprint.p(' ')
    goldText = attrtext.getGoldText(playerId)
    eprint.p(goldText)
    eprint.p('\n')
    attrprint.printHpAndMpBar(playerId)
    mainMenuText = textload.getTextData(textload.stageWordId,'68')
    eprint.sontitleprint(mainMenuText)
    eprint.p('\n')
    askForMainMenu = ans.optionint(ans.mainmenu,4,'left',askfor=False,cmdSize='center')
    cmdList = cmdList + askForMainMenu
    systemMenuText = textload.getTextData(textload.stageWordId,'69')
    eprint.sontitleprint(systemMenuText)
    eprint.p('\n')
    systemMenuStartId = len(askForMainMenu)
    askForSystemMenu = ans.optionint(ans.systemmenu,4,'left',askfor=False,cmdSize='center',startId=systemMenuStartId)
    cmdList = cmdList + askForSystemMenu
    return cmdList