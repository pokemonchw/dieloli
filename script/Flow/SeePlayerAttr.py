from script.Core import EraPrint,CacheContorl,PyCmd,GameInit,ValueHandle,TextLoading
from script.Design import PanelStateHandle,GameTime,CharacterHandle,AttrCalculation,MapHandle
from script.Panel import SeePlayerAttrPanel

panelList = ['PlayerMainAttrPanel','PlayerEquipmentPanel','PlayerItemPanel','PlayerExperiencePanel',
                   'PlayerLevelPanel','PlayerFeaturesPanel','PlayerEngravingPanel']

# 创建角色时用于查看角色属性的流程
def acknowledgmentAttribute_func():
    playerId = CacheContorl.playObject['objectId']
    AttrCalculation.setAttrOver(playerId)
    inputS = []
    attrInpurList = seeAttrInEveryTime_func()
    inputS = ValueHandle.listAppendToList(attrInpurList,inputS)
    flowReturn = SeePlayerAttrPanel.inputAttrOverPanel()
    inputS = ValueHandle.listAppendToList(flowReturn,inputS)
    acknowledgmentAttributeAns(inputS)

# 创建角色时用于查看角色属性的流程的事件控制
def acknowledgmentAttributeAns(inputList):
    playerId = CacheContorl.playObject['objectId']
    yrn = GameInit.askfor_All(inputList)
    showAttrHandleData = TextLoading.getTextData(TextLoading.cmdId,'seeAttrPanelHandle')
    PyCmd.clr_cmd()
    if yrn in panelList:
        PanelStateHandle.panelStateChange(yrn)
        acknowledgmentAttribute_func()
    elif yrn == '0':
        GameTime.initTime()
        AttrCalculation.setAttrOver(playerId)
        CharacterHandle.initCharacterList()
        SeePlayerAttrPanel.initShowAttrPanelList()
        playerPosition = CacheContorl.playObject['object'][playerId]['Position']
        MapHandle.playerMoveScene('0', playerPosition, playerId)
        from script.Flow import Main
        Main.mainFrame_func()
    elif yrn == '1':
        CacheContorl.wframeMouse['wFrameRePrint'] = 1
        EraPrint.pnextscreen()
        SeePlayerAttrPanel.initShowAttrPanelList()
        from script.Design import StartFlow
        StartFlow.main_func()
    elif yrn in showAttrHandleData:
        index = showAttrHandleData.index(yrn)
        index = str(index)
        CacheContorl.panelState['AttrShowHandlePanel'] = index
        acknowledgmentAttribute_func()

# 通用查看角色属性流程
def seeAttrOnEveryTime_func(oldPanel,tooOldFlow = None):
    objectId = CacheContorl.playObject['objectId']
    if oldPanel == 'InScenePanel':
        sceneId = CacheContorl.playObject['object']['0']['Position']
        objectIdList = MapHandle.getScenePlayerIdList(sceneId)
    else:
        objectIdList = ValueHandle.dictKeysToList(CacheContorl.playObject['object'])
    objectIdIndex = objectIdList.index(objectId)
    inputS = []
    seeAttrList = seeAttrInEveryTime_func()
    inputS = inputS + seeAttrList
    askSeeAttr = SeePlayerAttrPanel.askForSeeAttr()
    inputS = inputS + askSeeAttr
    yrn = GameInit.askfor_All(inputS)
    PyCmd.clr_cmd()
    showAttrHandleData = TextLoading.getTextData(TextLoading.cmdId, 'seeAttrPanelHandle')
    objectMax = objectIdList[len(objectIdList) - 1]
    if yrn in showAttrHandleData:
        index = showAttrHandleData.index(yrn)
        index = str(index)
        CacheContorl.panelState['AttrShowHandlePanel'] = index
        seeAttrOnEveryTime_func(oldPanel,tooOldFlow)
    elif yrn in panelList:
        PanelStateHandle.panelStateChange(yrn)
        seeAttrOnEveryTime_func(oldPanel,tooOldFlow)
    elif yrn == '0':
        if objectIdIndex == 0:
            CacheContorl.playObject['objectId'] = objectMax
            seeAttrOnEveryTime_func(oldPanel,tooOldFlow)
        else:
            playerId = objectIdList[objectIdIndex - 1]
            CacheContorl.playObject['objectId'] = playerId
            seeAttrOnEveryTime_func(oldPanel,tooOldFlow)
    elif yrn == '1':
        from script.Flow import Main,SeePlayerList,InScene
        if oldPanel == 'MainFramePanel':
            SeePlayerAttrPanel.initShowAttrPanelList()
            CacheContorl.playObject['objectId'] = '0'
            Main.mainFrame_func()
        elif oldPanel == 'SeePlayerListPanel':
            SeePlayerAttrPanel.initShowAttrPanelList()
            SeePlayerList.seePlayerList_func(tooOldFlow)
        elif oldPanel == 'InScenePanel':
            SeePlayerAttrPanel.initShowAttrPanelList()
            CacheContorl.playObject['objectId'] = '0'
            InScene.getInScene_func()
    elif yrn == '2':
        if objectId == objectMax:
            objectId = objectIdList[0]
            CacheContorl.playObject['objectId'] = objectId
            seeAttrOnEveryTime_func(oldPanel,tooOldFlow)
        else:
            objectId = objectIdList[objectIdIndex  + 1]
            CacheContorl.playObject['objectId'] = objectId
            seeAttrOnEveryTime_func(oldPanel,tooOldFlow)

# 用于任何时候查看角色属性的流程
def seeAttrInEveryTime_func():
    playerId = CacheContorl.playObject['objectId']
    showAttrHandle = CacheContorl.panelState['AttrShowHandlePanel']
    inputS = []
    playerMainAttrPanelAsk = SeePlayerAttrPanel.seePlayerMainAttrPanel(playerId)
    inputS.append(playerMainAttrPanelAsk)
    if showAttrHandle == '0':
        playerEquipmentPanelAsk = SeePlayerAttrPanel.seePlayerEquipmentPanel(playerId)
        inputS.append(playerEquipmentPanelAsk)
        playerItemPanelAsk = SeePlayerAttrPanel.seePlayerItemPanel(playerId)
        inputS.append(playerItemPanelAsk)
    elif showAttrHandle == '1':
        playerExperiencePanelAsk = SeePlayerAttrPanel.seePlayerExperiencePanel(playerId)
        inputS.append(playerExperiencePanelAsk)
        playerLevelPanelAsk = SeePlayerAttrPanel.seePlayerLevelPanel(playerId)
        inputS.append(playerLevelPanelAsk)
    elif showAttrHandle == '2':
        playerFeaturesPanelAsk = SeePlayerAttrPanel.seePlayerFeaturesPanel(playerId)
        inputS.append(playerFeaturesPanelAsk)
        playerEngravingPanelAsk = SeePlayerAttrPanel.seePlayerEngravingPanel(playerId)
        inputS.append(playerEngravingPanelAsk)
    EraPrint.pline()
    seeAttrPanelHandleAsk = SeePlayerAttrPanel.seeAttrShowHandlePanel()
    inputS = ValueHandle.listAppendToList(seeAttrPanelHandleAsk,inputS)
    return inputS
