from script.Core import EraPrint,CacheContorl,PyCmd,GameInit,ValueHandle,TextLoading,GameConfig
from script.Design import PanelStateHandle,GameTime,CharacterHandle,AttrCalculation,MapHandle
from script.Panel import SeeCharacterAttrPanel

panelList = ['CharacterMainAttrPanel','CharacterEquipmentPanel','CharacterItemPanel','CharacterExperiencePanel','CharacterLevelPanel','CharacterFeaturesPanel','CharacterEngravingPanel']

# 创建角色时用于查看角色属性的流程
def acknowledgmentAttribute_func():
    while(True):
        characterId = CacheContorl.characterData['characterId']
        AttrCalculation.setAttrOver(characterId)
        inputS = []
        attrInpurList = seeAttrInEveryTime_func()
        inputS = ValueHandle.listAppendToList(attrInpurList,inputS)
        flowReturn = SeeCharacterAttrPanel.inputAttrOverPanel()
        inputS = ValueHandle.listAppendToList(flowReturn,inputS)
        characterId = CacheContorl.characterData['characterId']
        yrn = GameInit.askfor_All(inputS)
        showAttrHandleData = TextLoading.getTextData(TextLoading.cmdPath,'seeAttrPanelHandle')
        PyCmd.clr_cmd()
        if yrn in panelList:
            PanelStateHandle.panelStateChange(yrn)
        elif yrn == '0':
            GameTime.initTime()
            AttrCalculation.setAttrOver(characterId)
            CharacterHandle.initCharacterList()
            SeeCharacterAttrPanel.initShowAttrPanelList()
            characterPosition = CacheContorl.characterData['character'][characterId]['Position']
            MapHandle.characterMoveScene(['0'], characterPosition, characterId)
            CacheContorl.nowFlowId = 'main'
            break
        elif yrn == '1':
            CacheContorl.wframeMouse['wFrameRePrint'] = 1
            EraPrint.pnextscreen()
            SeeCharacterAttrPanel.initShowAttrPanelList()
            CacheContorl.nowFlowId = 'title_frame'
            break
        elif yrn in showAttrHandleData:
            CacheContorl.panelState['AttrShowHandlePanel'] = str(showAttrHandleData.index(yrn))

# 通用查看角色属性流程
def seeAttrOnEveryTime_func():
    while(True):
        characterId = CacheContorl.characterData['characterId']
        if CacheContorl.oldFlowId == 'in_scene':
            sceneId = CacheContorl.characterData['character']['0']['Position']
            characterIdList = MapHandle.getSceneCharacterIdList(sceneId)
        else:
            characterIdList = ValueHandle.dictKeysToList(CacheContorl.characterData['character'])
        characterIdIndex = characterIdList.index(characterId)
        inputS = []
        seeAttrList = seeAttrInEveryTime_func()
        inputS = inputS + seeAttrList
        askSeeAttr = SeeCharacterAttrPanel.askForSeeAttr()
        inputS = inputS + askSeeAttr
        yrn = GameInit.askfor_All(inputS)
        PyCmd.clr_cmd()
        showAttrHandleData = TextLoading.getTextData(TextLoading.cmdPath, 'seeAttrPanelHandle')
        characterMax = characterIdList[len(characterIdList) - 1]
        if yrn in showAttrHandleData:
            index = showAttrHandleData.index(yrn)
            index = str(index)
            CacheContorl.panelState['AttrShowHandlePanel'] = index
        elif yrn in panelList:
            PanelStateHandle.panelStateChange(yrn)
        elif yrn == '0':
            if characterIdIndex == 0:
                CacheContorl.characterData['characterId'] = characterMax
            else:
                characterId = characterIdList[characterIdIndex - 1]
                CacheContorl.characterData['characterId'] = characterId
        elif yrn == '1':
            from script.Flow import SeeCharacterList
            SeeCharacterAttrPanel.initShowAttrPanelList()
            if CacheContorl.oldFlowId == 'main':
                CacheContorl.characterData['characterId'] = '0'
            elif CacheContorl.oldFlowId == 'see_character_list':
                characterListShow = int(GameConfig.characterlist_show)
                nowPageId = characterIdIndex / characterListShow
                CacheContorl.panelState['SeeCharacterListPanel'] = nowPageId
            CacheContorl.nowFlowId = CacheContorl.oldFlowId
            CacheContorl.oldFlowId = CacheContorl.tooOldFlowId
            break
        elif yrn == '2':
            if characterId == characterMax:
                characterId = characterIdList[0]
                CacheContorl.characterData['characterId'] = characterId
            else:
                characterId = characterIdList[characterIdIndex  + 1]
                CacheContorl.characterData['characterId'] = characterId

# 用于任何时候查看角色属性的流程
def seeAttrInEveryTime_func():
    characterId = CacheContorl.characterData['characterId']
    showAttrHandle = CacheContorl.panelState['AttrShowHandlePanel']
    inputS = []
    characterMainAttrPanelAsk = SeeCharacterAttrPanel.seeCharacterMainAttrPanel(characterId)
    inputS.append(characterMainAttrPanelAsk)
    if showAttrHandle == '0':
        characterEquipmentPanelAsk = SeeCharacterAttrPanel.seeCharacterEquipmentPanel(characterId)
        inputS.append(characterEquipmentPanelAsk)
        characterItemPanelAsk = SeeCharacterAttrPanel.seeCharacterItemPanel(characterId)
        inputS.append(characterItemPanelAsk)
    elif showAttrHandle == '1':
        characterExperiencePanelAsk = SeeCharacterAttrPanel.seeCharacterExperiencePanel(characterId)
        inputS.append(characterExperiencePanelAsk)
        characterLevelPanelAsk = SeeCharacterAttrPanel.seeCharacterLevelPanel(characterId)
        inputS.append(characterLevelPanelAsk)
    elif showAttrHandle == '2':
        characterFeaturesPanelAsk = SeeCharacterAttrPanel.seeCharacterFeaturesPanel(characterId)
        inputS.append(characterFeaturesPanelAsk)
        characterEngravingPanelAsk = SeeCharacterAttrPanel.seeCharacterEngravingPanel(characterId)
        inputS.append(characterEngravingPanelAsk)
    EraPrint.pline()
    seeAttrPanelHandleAsk = SeeCharacterAttrPanel.seeAttrShowHandlePanel()
    inputS = ValueHandle.listAppendToList(seeAttrPanelHandleAsk,inputS)
    return inputS
