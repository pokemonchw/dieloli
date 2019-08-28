from script.Core import EraPrint,CacheContorl,TextLoading,PyCmd
from script.Design import AttrHandle,AttrText,AttrPrint,GameTime,CmdButtonQueue

def mainFramePanel() -> list:
    '''
    游戏主菜单
    '''
    cmdList = []
    characterId = CacheContorl.characterData['characterId']
    characterData = AttrHandle.getAttrData(characterId)
    titleText = TextLoading.getTextData(TextLoading.stageWordPath, '64')
    EraPrint.plt(titleText)
    dateText = GameTime.getDateText()
    EraPrint.p(dateText)
    EraPrint.p(' ')
    weekDateText = GameTime.getWeekDayText()
    EraPrint.p(weekDateText)
    EraPrint.p(' ')
    characterName = characterData['Name']
    PyCmd.pcmd(characterName,characterName,None)
    cmdList.append(characterName)
    EraPrint.p(' ')
    goldText = AttrText.getGoldText(characterId)
    EraPrint.p(goldText)
    EraPrint.p('\n')
    AttrPrint.printHpAndMpBar(characterId)
    mainMenuText = TextLoading.getTextData(TextLoading.stageWordPath,'68')
    EraPrint.sontitleprint(mainMenuText)
    EraPrint.p('\n')
    askForMainMenu = CmdButtonQueue.optionint(CmdButtonQueue.mainmenu, 4, 'left', askfor=False, cmdSize='center')
    cmdList = cmdList + askForMainMenu
    systemMenuText = TextLoading.getTextData(TextLoading.stageWordPath,'69')
    EraPrint.sontitleprint(systemMenuText)
    EraPrint.p('\n')
    systemMenuStartId = len(askForMainMenu)
    askForSystemMenu = CmdButtonQueue.optionint(CmdButtonQueue.systemmenu, 4, 'left', askfor=False, cmdSize='center', startId=systemMenuStartId)
    cmdList = cmdList + askForSystemMenu
    return cmdList
