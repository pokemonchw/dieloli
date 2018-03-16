import script.AttrCalculation as attr
import core.EraPrint as eprint
import core.CacheContorl as cache
import script.Panel.SeePlayerAttrPanel as seeplayerattrpanel
import core.PyCmd as pycmd
import script.GameTime as gametime
import core.game as game
import script.PanelStateHandle as panelstatehandle
import core.ValueHandle as valuehandle
import core.TextLoading as textload

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
    ans = game.askfor_All(inputList)
    showAttrHandleData = textload.getTextData(textload.cmdId,'seeAttrPanelHandle')
    panelList = ['PlayerMainAttrPanel','PlayerEquipmentPanel','PlayerItemPanel','PlayerExperiencePanel',
                   'PlayerLevelPanel','PlayerFeaturesPanel','PlayerEngravingPanel']
    if ans in panelList:
        panelstatehandle.panelStateChange(ans)
        updateAcknowledg()
    elif ans == '0':
        pycmd.clr_cmd()
        gametime.initTime()
        seeplayerattrpanel.initShowAttrPanelList()
        import script.flow.MainFrame as mainframe
        mainframe.mainFrame_func()
    elif ans == '1':
        cache.wframeMouse['wFrameRePrint'] = 1
        eprint.pnextscreen()
        seeplayerattrpanel.initShowAttrPanelList()
        import script.mainflow as mainflow
        mainflow.main_func()
    elif ans == showAttrHandleData[0]:
        pycmd.clr_cmd()
        cache.panelState['AttrShowHandlePanel'] = '0'
        updateAcknowledg()
    elif ans == showAttrHandleData[1]:
        pycmd.clr_cmd()
        cache.panelState['AttrShowHandlePanel'] = '1'
        updateAcknowledg()
    elif ans == showAttrHandleData[2]:
        pycmd.clr_cmd()
        cache.panelState['AttrShowHandlePanel'] = '2'
        updateAcknowledg()
    pass

# 用于刷新创建角色时查看角色属性的流程面板
def updateAcknowledg():
    pycmd.clr_cmd()
    acknowledgmentAttribute_func()
    pass

# 用于任何时候查看角色属性的流程
def seeAttrInEveryTime_func():
    playerId = cache.playObject['objectId']
    showAttrHandle = cache.panelState['AttrShowHandlePanel']
    attr.setAttrOver(playerId)
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