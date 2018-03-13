import script.AttrCalculation as attr
import core.EraPrint as eprint
import core.CacheContorl as cache
import script.Panel.SeePlayerAttrPanel as seeplayerattrpanel
import core.PyCmd as pycmd
import script.GameTime as gametime
import core.game as game
import script.PanelStateHandle as panelstatehandle
import core.ValueHandle as valuehandle

def acknowledgmentAttribute_func():
    playerId = cache.playObject['objectId']
    attr.setAttrOver(playerId)
    inputS = []
    playerMainAttrPanelAsk = seeplayerattrpanel.seePlayerMainAttrPanel(playerId)
    inputS.append(playerMainAttrPanelAsk)
    playerEquipmentPanelAsk = seeplayerattrpanel.seePlayerEquipmentPanel(playerId)
    inputS.append(playerEquipmentPanelAsk)
    playerItemPanelAsk = seeplayerattrpanel.seePlayerItemPanel(playerId)
    inputS.append(playerItemPanelAsk)
    playerExperiencePanelAsk = seeplayerattrpanel.seePlayerExperiencePanel(playerId)
    inputS.append(playerExperiencePanelAsk)
    playerLevelPanelAsk = seeplayerattrpanel.seePlayerLevelPanel(playerId)
    inputS.append(playerLevelPanelAsk)
    playerFeaturesPanelAsk = seeplayerattrpanel.seePlayerFeaturesPanel(playerId)
    inputS.append(playerFeaturesPanelAsk)
    playerEngravingPanelAsk = seeplayerattrpanel.seePlayerEngravingPanel(playerId)
    inputS.append(playerEngravingPanelAsk)
    eprint.pline()
    flowReturn = seeplayerattrpanel.inputAttrOverPanel()
    inputS = valuehandle.listAppendToList(flowReturn,inputS)
    acknowledgmentAttributeAns(inputS)
    pass

def acknowledgmentAttributeAns(inputList):
    ans = game.askfor_All(inputList)
    panelList = ['PlayerMainAttrPanel','PlayerEquipmentPanel','PlayerItemPanel','PlayerExperiencePanel',
                   'PlayerLevelPanel','PlayerFeaturesPanel','PlayerEngravingPanel']
    if ans in panelList:
        panelstatehandle.panelStateChange(ans)
        updateAcknowledg()
    elif ans == '0':
        pycmd.clr_cmd()
        gametime.initTime()
        import script.flow.MainFrame as mainframe
        mainframe.mainFrame_func()
    elif ans == '1':
        cache.wframeMouse['wFrameRePrint'] = 1
        eprint.pnextscreen()
        import script.mainflow as mainflow
        mainflow.main_func()
    pass

def updateAcknowledg():
    pycmd.clr_cmd()
    acknowledgmentAttribute_func()
    pass