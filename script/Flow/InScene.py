from script.Core import CacheContorl,GameInit,PyCmd,GameConfig
from script.Design import MapHandle,CharacterHandle
from script.Panel import InScenePanel
from script.Flow import SeeCharacterAttr
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
    if len(sceneCharacterList) > 1:
        CacheContorl.characterData['characterId'] = sceneCharacterList[0]
        seeScene_func('0')
    else:
        seeScene_func('1')

# 用于查看当前场景的流程
def seeScene_func(judge):
    inputS = []
    time1 = datetime.datetime.now()
    InScenePanel.seeScenePanel()
    time2 = datetime.datetime.now()
    print(time2-time1)
    if judge  == '0':
        inputS = inputS + InScenePanel.seeSceneCharacterListPanel()
    inSceneCmdList1 = InScenePanel.changeSceneCharacterListPanel()
    startId1 = len(inSceneCmdList1)
    scenePath = CacheContorl.characterData['character']['0']['Position']
    sceneCharacterNameList = MapHandle.getSceneCharacterNameList(scenePath)
    if len(sceneCharacterNameList) == 1:
        CacheContorl.characterData['characterId'] = '0'
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
        seeScene_func(judge)
    elif yrn == inSceneCmdList1[0]:
        if int(nowPage) == 0:
            CacheContorl.panelState['SeeSceneCharacterListPanel'] = pageMax
        else:
            CacheContorl.panelState['SeeSceneCharacterListPanel'] = int(nowPage) - 1
        seeScene_func(judge)
    elif yrn == inSceneCmdList1[1]:
        if int(nowPage) == pageMax:
            CacheContorl.panelState['SeeSceneCharacterListPanel'] = 0
        else:
            CacheContorl.panelState['SeeSceneCharacterListPanel'] = int(nowPage) + 1
        seeScene_func(judge)
    elif yrn == inSceneCmdList2[0]:
        from script.Flow import SeeMap
        nowMap = MapHandle.getMapForPath(CacheContorl.characterData['character']['0']['Position'])
        CacheContorl.nowMap = nowMap
        SeeMap.seeMapFlow()
    elif yrn == inSceneCmdList2[1]:
        SeeCharacterAttr.seeAttrOnEveryTime_func('InScenePanel')
    elif yrn == inSceneCmdList2[2]:
        CacheContorl.characterData['characterId'] = '0'
        SeeCharacterAttr.seeAttrOnEveryTime_func('InScenePanel')
