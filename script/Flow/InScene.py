from script.Core import CacheContorl,GameInit,PyCmd,GameConfig
from script.Design import MapHandle,CharacterHandle,PanelStateHandle
from script.Panel import InScenePanel
import math,datetime

# 用于进入场景流程
def getInScene_func():
    PyCmd.clr_cmd()
    scenePath = CacheContorl.characterData['character']['0']['Position']
    MapHandle.sortSceneCharacterId(scenePath)
    CacheContorl.nowMap = MapHandle.getMapForPath(scenePath)
    scenePathStr = MapHandle.getMapSystemPathStrForList(scenePath)
    sceneData = CacheContorl.sceneData[scenePathStr].copy()
    sceneCharacterList = sceneData['SceneCharacterData']
    if '0' not in sceneCharacterList:
        characterIdList = ['0']
        sceneCharacterList = sceneCharacterList + characterIdList
        CacheContorl.sceneData[scenePathStr]['SceneCharacterData'] = sceneCharacterList
    if len(sceneCharacterList) > 1 and CacheContorl.characterData['characterId'] == '0':
        nowNameList = MapHandle.getSceneCharacterNameList(scenePath)
        nowNameList.remove(CacheContorl.characterData['character']['0']['Name'])
        CacheContorl.characterData['characterId'] = MapHandle.getCharacterIdByCharacterName(nowNameList[0],scenePath)
        if CacheContorl.oldCharacterId != '0':
            CacheContorl.characterData['characterId'] = CacheContorl.oldCharacterId
            CacheContorl.oldCharacterId = '0'
    if len(sceneCharacterList) > 1:
        seeScene_func('0')
    else:
        seeScene_func('1')

# 用于查看当前场景的流程
def seeScene_func(judge):
    while(True):
        inputS = []
        time1 = datetime.datetime.now()
        InScenePanel.seeScenePanel()
        scenePath = CacheContorl.characterData['character']['0']['Position']
        sceneCharacterNameList = MapHandle.getSceneCharacterNameList(scenePath)
        if len(sceneCharacterNameList) == 1:
            CacheContorl.characterData['characterId'] = '0'
        inSceneCmdList1 = []
        if judge == '0':
            if CacheContorl.panelState['SeeSceneCharacterListPage'] == '0':
                inputS = inputS + InScenePanel.seeSceneCharacterListPanel()
                inSceneCmdList1 = InScenePanel.changeSceneCharacterListPanel()
            inputS.append('SeeSceneCharacterListPage')
        startId1 = len(inSceneCmdList1)
        InScenePanel.seeCharacterInfoPanel()
        inSceneCmdList2 = InScenePanel.inSceneButtonPanel(startId1)
        inputS = inputS + inSceneCmdList1 + inSceneCmdList2
        yrn = GameInit.askfor_All(inputS)
        PyCmd.clr_cmd()
        nowPage = int(CacheContorl.panelState['SeeSceneCharacterListPanel'])
        nameListMax = int(GameConfig.in_scene_see_player_max)
        characterMax = CharacterHandle.getCharacterIndexMax() - 1
        pageMax = math.floor(characterMax / nameListMax)
        if yrn in sceneCharacterNameList:
            CacheContorl.characterData['characterId'] = MapHandle.getCharacterIdByCharacterName(yrn,scenePath)
        elif judge == '0' and yrn not in inSceneCmdList2 and yrn != 'SeeSceneCharacterListPage':
            if yrn == inSceneCmdList1[0]:
                CacheContorl.panelState['SeeSceneCharacterListPanel'] = 0
            elif yrn == inSceneCmdList1[1]:
                if int(nowPage) == 0:
                    CacheContorl.panelState['SeeSceneCharacterListPanel'] = pageMax
                else:
                    CacheContorl.panelState['SeeSceneCharacterListPanel'] = int(nowPage) - 1
            elif yrn == inSceneCmdList1[2]:
                CacheContorl.panelState['SeeSceneCharacterListPanel'] = InScenePanel.jumpCharacterListPagePanel()
            elif yrn == inSceneCmdList1[3]:
                if int(nowPage) == pageMax:
                    CacheContorl.panelState['SeeSceneCharacterListPanel'] = 0
                else:
                    CacheContorl.panelState['SeeSceneCharacterListPanel'] = int(nowPage) + 1
            elif yrn == inSceneCmdList1[4]:
                CacheContorl.panelState['SeeSceneCharacterListPanel'] = pageMax
        elif yrn == inSceneCmdList2[0]:
            CacheContorl.nowFlowId = 'see_map'
            nowMap = MapHandle.getMapForPath(CacheContorl.characterData['character']['0']['Position'])
            CacheContorl.nowMap = nowMap
            break
        elif yrn in [inSceneCmdList2[1],inSceneCmdList2[2]]:
            if yrn == inSceneCmdList2[2]:
                CacheContorl.oldCharacterId = CacheContorl.characterData['characterId']
                CacheContorl.characterData['characterId'] = '0'
            CacheContorl.nowFlowId = 'see_character_attr'
            CacheContorl.oldFlowId = 'in_scene'
            break
        elif yrn == 'SeeSceneCharacterListPage':
            PanelStateHandle.panelStateChange(yrn)
