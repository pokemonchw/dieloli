from script.Core import EraPrint,CacheContorl,PyCmd,GameInit,TextLoading,GameConfig
from script.Design import PanelStateHandle,AttrCalculation,MapHandle,CharacterHandle
from script.Panel import SeeCharacterAttrPanel
import math
from script.Flow import GameStartFlow

panelList = ['CharacterMainAttrPanel','CharacterEquipmentPanel','CharacterItemPanel','CharacterExperiencePanel','CharacterLevelPanel','CharacterFeaturesPanel','CharacterEngravingPanel']

def acknowledgmentAttribute_func():
    '''
    创建角色时用于查看角色属性的流程
    '''
    while(True):
        characterId = CacheContorl.characterData['characterId']
        AttrCalculation.setAttrOver(characterId)
        CharacterHandle.initCharacterList()
        inputS = []
        seeAttrInEveryTime_func()
        flowReturn = SeeCharacterAttrPanel.inputAttrOverPanel()
        inputS = flowReturn + inputS
        characterId = CacheContorl.characterData['characterId']
        yrn = GameInit.askfor_All(inputS)
        showAttrHandleData = TextLoading.getTextData(TextLoading.cmdPath,'seeAttrPanelHandle')
        PyCmd.clr_cmd()
        if yrn in panelList:
            PanelStateHandle.panelStateChange(yrn)
        elif yrn == '0':
            GameStartFlow.initGameStart()
            break
        elif yrn == '1':
            CacheContorl.wframeMouse['wFrameRePrint'] = 1
            EraPrint.pnextscreen()
            SeeCharacterAttrPanel.initShowAttrPanelList()
            CacheContorl.nowFlowId = 'title_frame'
            break
        elif yrn in showAttrHandleData:
            CacheContorl.panelState['AttrShowHandlePanel'] = str(showAttrHandleData.index(yrn))

def seeAttrOnEveryTime_func():
    '''
    通用用于查看角色属性的流程
    '''
    while(True):
        characterId = CacheContorl.characterData['characterId']
        if CacheContorl.oldFlowId == 'in_scene':
            nowScene = CacheContorl.characterData['character']['0']['Position']
            nowSceneStr = MapHandle.getMapSystemPathStrForList(nowScene)
            characterIdList = MapHandle.getSceneCharacterIdList(nowSceneStr)
        else:
            characterIdList = list(CacheContorl.characterData['character'].keys())
        characterIdIndex = characterIdList.index(characterId)
        inputS = []
        seeAttrInEveryTime_func()
        askSeeAttr = SeeCharacterAttrPanel.askForSeeAttr()
        inputS += askSeeAttr
        inputs1 = SeeCharacterAttrPanel.askForSeeAttrCmd()
        inputS += inputs1
        yrn = GameInit.askfor_All(inputS)
        PyCmd.clr_cmd()
        showAttrHandleData = TextLoading.getTextData(TextLoading.cmdPath, 'seeAttrPanelHandle')
        characterMax = characterIdList[len(characterIdList) - 1]
        if yrn in showAttrHandleData:
            CacheContorl.panelState['AttrShowHandlePanel'] = yrn
        elif yrn == '0':
            if characterIdIndex == 0:
                CacheContorl.characterData['characterId'] = characterMax
            else:
                characterId = characterIdList[characterIdIndex - 1]
                CacheContorl.characterData['characterId'] = characterId
        elif yrn == '1':
            if CacheContorl.oldFlowId == 'main':
                CacheContorl.characterData['characterId'] = '0'
            elif CacheContorl.oldFlowId == 'see_character_list':
                characterListShow = int(GameConfig.characterlist_show)
                nowPageId = characterIdIndex / characterListShow
                CacheContorl.panelState['SeeCharacterListPanel'] = nowPageId
            elif CacheContorl.oldFlowId == 'in_scene':
                scenePath = CacheContorl.characterData['character']['0']['Position']
                scenePathStr = MapHandle.getMapSystemPathStrForList(scenePath)
                nameList = MapHandle.getSceneCharacterNameList(scenePathStr,True)
                nowCharacterName = CacheContorl.characterData['character'][CacheContorl.characterData['characterId']]['Name']
                try:
                    nowCharacterIndex = nameList.index(nowCharacterName)
                except ValueError:
                    nowCharacterIndex = 0
                nameListMax = int(GameConfig.in_scene_see_player_max)
                nowSceneCharacterListPage = math.floor(nowCharacterIndex / nameListMax)
                CacheContorl.panelState['SeeSceneCharacterListPanel'] = nowSceneCharacterListPage
            CacheContorl.panelState['AttrShowHandlePanel'] = 'MainAttr'
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

def seeAttrInEveryTime_func():
    '''
    用于在任何时候查看角色属性的流程
    '''
    characterId = CacheContorl.characterData['characterId']
    nowAttrPanel = CacheContorl.panelState['AttrShowHandlePanel']
    return SeeCharacterAttrPanel.panelData[nowAttrPanel](characterId)
