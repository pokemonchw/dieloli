from script.Core import CacheContorl,GameInit,PyCmd
from script.Design import MapHandle
from script.Panel import InScenePanel
from script.Flow import SeeCharacterAttr

# 用于进入场景流程
def getInScene_func():
    PyCmd.clr_cmd()
    scenePath = CacheContorl.characterData['character']['0']['Position']
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
    InScenePanel.seeScenePanel()
    if judge  == '0':
        inputS = inputS + InScenePanel.seeSceneCharacterListPanel()
    else:
        pass
    scenePath = CacheContorl.characterData['character']['0']['Position']
    sceneCharacterNameList = MapHandle.getSceneCharacterNameList(scenePath)
    if len(sceneCharacterNameList) == 1:
        CacheContorl.characterData['characterId'] = '0'
    else:
        pass
    InScenePanel.seeCharacterInfoPanel()
    inSceneCmdList1 = InScenePanel.inSceneButtonPanel()
    inputS = inputS + inSceneCmdList1
    startId1 = len(inSceneCmdList1)
    yrn = GameInit.askfor_All(inputS)
    PyCmd.clr_cmd()
    if yrn in sceneCharacterNameList:
        CacheContorl.characterData['characterId'] = MapHandle.getCharacterIdByCharacterName(yrn,scenePath)
        seeScene_func(judge)
    elif yrn == '0':
        from script.Flow import SeeMap
        nowMap = MapHandle.getMapForPath(CacheContorl.characterData['character']['0']['Position'])
        CacheContorl.nowMap = nowMap
        SeeMap.seeMapFlow()
    elif yrn == '1':
        SeeCharacterAttr.seeAttrOnEveryTime_func('InScenePanel')
    elif yrn == '2':
        CacheContorl.characterData['characterId'] = '0'
        SeeCharacterAttr.seeAttrOnEveryTime_func('InScenePanel')
