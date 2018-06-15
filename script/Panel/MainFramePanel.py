from core import EraPrint,CacheContorl,TextLoading,PyCmd
from design import AttrHandle,AttrText,AttrPrint,GameTime,Ans

# 游戏主页流程
def mainFramePanel():
    cmdList = []
    playerId = CacheContorl.playObject['objectId']
    playerData = AttrHandle.getAttrData(playerId)
    titleText = TextLoading.getTextData(TextLoading.stageWordId, '64')
    EraPrint.plt(titleText)
    dateText = GameTime.getDateText()
    EraPrint.p(dateText)
    EraPrint.p(' ')
    weekDateText = GameTime.getWeekDayText()
    EraPrint.p(weekDateText)
    EraPrint.p(' ')
    playerName = playerData['Name']
    PyCmd.pcmd(playerName,playerName,None)
    cmdList.append(playerName)
    EraPrint.p(' ')
    goldText = AttrText.getGoldText(playerId)
    EraPrint.p(goldText)
    EraPrint.p('\n')
    AttrPrint.printHpAndMpBar(playerId)
    mainMenuText = TextLoading.getTextData(TextLoading.stageWordId,'68')
    EraPrint.sontitleprint(mainMenuText)
    EraPrint.p('\n')
    askForMainMenu = Ans.optionint(Ans.mainmenu, 4, 'left', askfor=False, cmdSize='center')
    cmdList = cmdList + askForMainMenu
    systemMenuText = TextLoading.getTextData(TextLoading.stageWordId,'69')
    EraPrint.sontitleprint(systemMenuText)
    EraPrint.p('\n')
    systemMenuStartId = len(askForMainMenu)
    askForSystemMenu = Ans.optionint(Ans.systemmenu, 4, 'left', askfor=False, cmdSize='center', startId=systemMenuStartId)
    cmdList = cmdList + askForSystemMenu
    return cmdList