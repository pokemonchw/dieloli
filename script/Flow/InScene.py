from script.Core import CacheContorl,GameInit,PyCmd
from script.Design import MapHandle
from script.Panel import InScenePanel
from script.Flow import SeePlayerAttr

# 用于进入场景流程
def getInScene_func():
    PyCmd.clr_cmd()
    scenePath = CacheContorl.playObject['object']['0']['Position']
    CacheContorl.nowMap = MapHandle.getMapForPath(scenePath)
    scenePathStr = MapHandle.getMapSystemPathStrForList(scenePath)
    sceneData = CacheContorl.sceneData[scenePathStr].copy()
    scenePlayerList = sceneData['ScenePlayerData']
    if '0' not in scenePlayerList:
        objectIdList = ['0']
        scenePlayerList = scenePlayerList + objectIdList
        CacheContorl.sceneData[scenePathStr]['ScenePlayerData'] = scenePlayerList
    if len(scenePlayerList) > 1:
        CacheContorl.playObject['objectId'] = scenePlayerList[0]
        seeScene_func('0')
    else:
        seeScene_func('1')

# 用于查看当前场景的流程
def seeScene_func(judge):
    inputS = []
    InScenePanel.seeScenePanel()
    if judge  == '0':
        inputS = inputS + InScenePanel.seeScenePlayerListPanel()
    else:
        pass
    scenePath = CacheContorl.playObject['object']['0']['Position']
    scenePlayerNameList = MapHandle.getScenePlayerNameList(scenePath)
    if len(scenePlayerNameList) == 1:
        CacheContorl.playObject['objectId'] = '0'
    else:
        pass
    InScenePanel.seeObjectInfoPanel()
    inSceneCmdList1 = InScenePanel.inSceneButtonPanel()
    inputS = inputS + inSceneCmdList1
    startId1 = len(inSceneCmdList1)
    yrn = GameInit.askfor_All(inputS)
    PyCmd.clr_cmd()
    if yrn in scenePlayerNameList:
        CacheContorl.playObject['objectId'] = MapHandle.getPlayerIdByPlayerName(yrn,scenePath)
        seeScene_func(judge)
    elif yrn == '0':
        from script.Flow import SeeMap
        nowMap = MapHandle.getMapForPath(CacheContorl.playObject['object']['0']['Position'])
        CacheContorl.nowMap = nowMap
        SeeMap.seeMapFlow()
    elif yrn == '1':
        SeePlayerAttr.seeAttrOnEveryTime_func('InScenePanel')
    elif yrn == '2':
        CacheContorl.playObject['objectId'] = '0'
        SeePlayerAttr.seeAttrOnEveryTime_func('InScenePanel')
