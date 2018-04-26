import design.AttrCalculation as attr
import core.EraPrint as eprint
import core.CacheContorl as cache
import Panel.SeePlayerAttrPanel as seeplayerattrpanel
import core.PyCmd as pycmd
import core.game as game
import core.ValueHandle as valuehandle
import core.TextLoading as textload
import design.PanelStateHandle as panelstatehandle
import design.GameTime as gametime
import design.CharacterHandle as characterhandle
import design.MapHandle as maphandle

panelList = ['PlayerMainAttrPanel','PlayerEquipmentPanel','PlayerItemPanel','PlayerExperiencePanel',
                   'PlayerLevelPanel','PlayerFeaturesPanel','PlayerEngravingPanel']

# 创建角色时用于查看角色属性的流程
def acknowledgmentAttribute_func():
    playerId = cache.playObject['objectId']
    attr.setAttrOver(playerId)
    inputS = []
    attrInpurList = seeAttrInEveryTime_func()
    inputS = valuehandle.listAppendToList(attrInpurList,inputS)
    flowReturn = seeplayerattrpanel.inputAttrOverPanel()
    inputS = valuehandle.listAppendToList(flowReturn,inputS)
    acknowledgmentAttributeAns(inputS)
    pass

# 创建角色时用于查看角色属性的流程的事件控制
def acknowledgmentAttributeAns(inputList):
    playerId = cache.playObject['objectId']
    yrn = game.askfor_All(inputList)
    showAttrHandleData = textload.getTextData(textload.cmdId,'seeAttrPanelHandle')
    pycmd.clr_cmd()
    if yrn in panelList:
        panelstatehandle.panelStateChange(yrn)
        acknowledgmentAttribute_func()
    elif yrn == '0':
        gametime.initTime()
        attr.setAttrOver(playerId)
        characterhandle.initCharacterList()
        seeplayerattrpanel.initShowAttrPanelList()
        playerPosition = cache.playObject['object'][playerId]['Position']
        maphandle.playerMoveScene('0', playerPosition, playerId)
        import script.flow.MainFrame as mainframe
        mainframe.mainFrame_func()
    elif yrn == '1':
        cache.wframeMouse['wFrameRePrint'] = 1
        eprint.pnextscreen()
        seeplayerattrpanel.initShowAttrPanelList()
        import design.mainflow as mainflow
        mainflow.main_func()
    elif yrn in showAttrHandleData:
        index = showAttrHandleData.index(yrn)
        index = str(index)
        cache.panelState['AttrShowHandlePanel'] = index
        acknowledgmentAttribute_func()

# 通用查看角色属性流程
def seeAttrOnEveryTime_func(oldPanel,tooOldFlow = None):
    playerId = cache.playObject['objectId']
    playerId = int(playerId)
    playerIdMax = characterhandle.getCharacterIndexMax()
    inputS = []
    seeAttrList = seeAttrInEveryTime_func()
    inputS = inputS + seeAttrList
    askSeeAttr = seeplayerattrpanel.askForSeeAttr()
    inputS = inputS + askSeeAttr
    yrn = game.askfor_All(inputS)
    pycmd.clr_cmd()
    showAttrHandleData = textload.getTextData(textload.cmdId, 'seeAttrPanelHandle')
    if yrn in showAttrHandleData:
        index = showAttrHandleData.index(yrn)
        index = str(index)
        cache.panelState['AttrShowHandlePanel'] = index
        seeAttrOnEveryTime_func(oldPanel,tooOldFlow)
    elif yrn in panelList:
        panelstatehandle.panelStateChange(yrn)
        seeAttrOnEveryTime_func(oldPanel,tooOldFlow)
    elif yrn == '0':
        if playerId == 0:
            playerIdMax = str(playerIdMax)
            cache.playObject['objectId'] = playerIdMax
            seeAttrOnEveryTime_func(oldPanel,tooOldFlow)
        else:
            playerId = str(playerId - 1)
            cache.playObject['objectId'] = playerId
            seeAttrOnEveryTime_func(oldPanel,tooOldFlow)
    elif yrn == '1':
        if oldPanel == 'MainFramePanel':
            import script.flow.MainFrame as mainframe
            seeplayerattrpanel.initShowAttrPanelList()
            cache.playObject['objectId'] = '0'
            mainframe.mainFrame_func()
        elif oldPanel == 'SeePlayerListPanel':
            seeplayerattrpanel.initShowAttrPanelList()
            import script.flow.SeePlayerList as seeplayerlist
            seeplayerlist.seePlayerList_func(tooOldFlow)
    elif yrn == '2':
        if playerId == playerIdMax:
            playerId = '0'
            cache.playObject['objectId'] = playerId
            seeAttrOnEveryTime_func(oldPanel,tooOldFlow)
        else:
            playerId = str(playerId + 1)
            cache.playObject['objectId'] = playerId
            seeAttrOnEveryTime_func(oldPanel,tooOldFlow)
    pass

# 用于任何时候查看角色属性的流程
def seeAttrInEveryTime_func():
    playerId = cache.playObject['objectId']
    showAttrHandle = cache.panelState['AttrShowHandlePanel']
    inputS = []
    playerMainAttrPanelAsk = seeplayerattrpanel.seePlayerMainAttrPanel(playerId)
    inputS.append(playerMainAttrPanelAsk)
    if showAttrHandle == '0':
        playerEquipmentPanelAsk = seeplayerattrpanel.seePlayerEquipmentPanel(playerId)
        inputS.append(playerEquipmentPanelAsk)
        playerItemPanelAsk = seeplayerattrpanel.seePlayerItemPanel(playerId)
        inputS.append(playerItemPanelAsk)
        pass
    elif showAttrHandle == '1':
        playerExperiencePanelAsk = seeplayerattrpanel.seePlayerExperiencePanel(playerId)
        inputS.append(playerExperiencePanelAsk)
        playerLevelPanelAsk = seeplayerattrpanel.seePlayerLevelPanel(playerId)
        inputS.append(playerLevelPanelAsk)
    elif showAttrHandle == '2':
        playerFeaturesPanelAsk = seeplayerattrpanel.seePlayerFeaturesPanel(playerId)
        inputS.append(playerFeaturesPanelAsk)
        playerEngravingPanelAsk = seeplayerattrpanel.seePlayerEngravingPanel(playerId)
        inputS.append(playerEngravingPanelAsk)
    eprint.pline()
    seeAttrPanelHandleAsk = seeplayerattrpanel.seeAttrShowHandlePanel()
    inputS = valuehandle.listAppendToList(seeAttrPanelHandleAsk,inputS)
    return inputS