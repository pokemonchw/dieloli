from script.Core import EraPrint,CacheContorl,PyCmd,GameInit,ValueHandle,TextLoading,GameConfig
from script.Design import PanelStateHandle,GameTime,CharacterHandle,AttrCalculation,MapHandle
from script.Panel import SeeCharacterAttrPanel

panelList = ['CharacterMainAttrPanel','CharacterEquipmentPanel','CharacterItemPanel','CharacterExperiencePanel',
                   'CharacterLevelPanel','CharacterFeaturesPanel','CharacterEngravingPanel']

# 创建角色时用于查看角色属性的流程
def acknowledgmentAttribute_func():
    characterId = CacheContorl.characterData['characterId']
    AttrCalculation.setAttrOver(characterId)
    inputS = []
    attrInpurList = seeAttrInEveryTime_func()
    inputS = ValueHandle.listAppendToList(attrInpurList,inputS)
    flowReturn = SeeCharacterAttrPanel.inputAttrOverPanel()
    inputS = ValueHandle.listAppendToList(flowReturn,inputS)
    acknowledgmentAttributeAns(inputS)

# 创建角色时用于查看角色属性的流程的事件控制
def acknowledgmentAttributeAns(inputList):
    characterId = CacheContorl.characterData['characterId']
    yrn = GameInit.askfor_All(inputList)
    showAttrHandleData = TextLoading.getTextData(TextLoading.cmdPath,'seeAttrPanelHandle')
    PyCmd.clr_cmd()
    if yrn in panelList:
        PanelStateHandle.panelStateChange(yrn)
        acknowledgmentAttribute_func()
    elif yrn == '0':
        GameTime.initTime()
        AttrCalculation.setAttrOver(characterId)
        CharacterHandle.initCharacterList()
        SeeCharacterAttrPanel.initShowAttrPanelList()
        characterPosition = CacheContorl.characterData['character'][characterId]['Position']
        MapHandle.characterMoveScene(['0'], characterPosition, characterId)
        from script.Flow import Main
        Main.mainFrame_func()
    elif yrn == '1':
        CacheContorl.wframeMouse['wFrameRePrint'] = 1
        EraPrint.pnextscreen()
        SeeCharacterAttrPanel.initShowAttrPanelList()
        from script.Design import StartFlow
        StartFlow.main_func()
    elif yrn in showAttrHandleData:
        index = showAttrHandleData.index(yrn)
        index = str(index)
        CacheContorl.panelState['AttrShowHandlePanel'] = index
        acknowledgmentAttribute_func()

# 通用查看角色属性流程
def seeAttrOnEveryTime_func(oldPanel,tooOldFlow = None):
    characterId = CacheContorl.characterData['characterId']
    if oldPanel == 'InScenePanel':
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
        seeAttrOnEveryTime_func(oldPanel,tooOldFlow)
    elif yrn in panelList:
        PanelStateHandle.panelStateChange(yrn)
        seeAttrOnEveryTime_func(oldPanel,tooOldFlow)
    elif yrn == '0':
        if characterIdIndex == 0:
            CacheContorl.characterData['characterId'] = characterMax
            seeAttrOnEveryTime_func(oldPanel,tooOldFlow)
        else:
            characterId = characterIdList[characterIdIndex - 1]
            CacheContorl.characterData['characterId'] = characterId
            seeAttrOnEveryTime_func(oldPanel,tooOldFlow)
    elif yrn == '1':
        from script.Flow import Main,SeeCharacterList,InScene
        if oldPanel == 'MainFramePanel':
            SeeCharacterAttrPanel.initShowAttrPanelList()
            CacheContorl.characterData['characterId'] = '0'
            Main.mainFrame_func()
        elif oldPanel == 'SeeCharacterListPanel':
            characterListShow = int(GameConfig.characterlist_show)
            nowPageId = characterIdIndex / characterListShow
            CacheContorl.panelState['SeeCharacterListPanel'] = nowPageId
            SeeCharacterAttrPanel.initShowAttrPanelList()
            SeeCharacterList.seeCharacterList_func(tooOldFlow)
        elif oldPanel == 'InScenePanel':
            SeeCharacterAttrPanel.initShowAttrPanelList()
            CacheContorl.characterData['characterId'] = '0'
            InScene.getInScene_func()
    elif yrn == '2':
        if characterId == characterMax:
            characterId = characterIdList[0]
            CacheContorl.characterData['characterId'] = characterId
            seeAttrOnEveryTime_func(oldPanel,tooOldFlow)
        else:
            characterId = characterIdList[characterIdIndex  + 1]
            CacheContorl.characterData['characterId'] = characterId
            seeAttrOnEveryTime_func(oldPanel,tooOldFlow)

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
