from script.Core import FlowHandle,CacheContorl,PyCmd
from script.Design import ObjectMove,MapHandle
from script.Panel import SeeMapPanel
import os

def seeMapFlow():
    PyCmd.clr_cmd()
    inputS = []
    mapCmd = SeeMapPanel.seeMapPanel()
    startId1 = len(mapCmd)
    inputS = inputS + mapCmd
    movePathCmdData = SeeMapPanel.seeMovePathPanel()
    movePathCmd = movePathCmdData['inputS']
    movePathList = movePathCmdData['scenePathList']
    seeMapCmd = SeeMapPanel.backScenePanel(startId1)
    inputS = inputS + seeMapCmd + movePathCmd
    yrn = FlowHandle.askfor_All(inputS)
    backButton = str(startId1)
    nowPosition = CacheContorl.playObject['object']['0']['Position']
    nowPositionMapId = MapHandle.getMapIdForScene(nowPosition)
    upMapButton = 'Null'
    downMapButton = 'Null'
    if str(nowPositionMapId) != '0' and str(CacheContorl.nowMapId) != '0':
        upMapButton = str(int(startId1) + 1)
    if str(nowPositionMapId) != str(CacheContorl.nowMapId):
        if upMapButton == 'Null':
            downMapButton = str(int(startId1) + 1)
        else:
            downMapButton = str(int(startId1) + 2)
    mapId = CacheContorl.nowMapId
    if yrn in mapCmd:
        nowTargetPath = MapHandle.getScenePathForMapSceneId(mapId,yrn)
        ObjectMove.playerMove(nowTargetPath)
    elif yrn == backButton:
        CacheContorl.nowMapId = '0'
        import script.Flow.InScene as inscene
        inscene.getInScene_func()
    elif yrn in movePathCmd:
        moveListId = movePathCmd.index(yrn)
        moveId = movePathList[moveListId]
        nowTargetPath = MapHandle.getScenePathForMapSceneId(mapId,moveId)
        ObjectMove.playerMove(nowTargetPath)
    elif upMapButton != 'Null' and yrn == upMapButton:
        nowMapPath = MapHandle.getPathForMapId(CacheContorl.nowMapId)
        upMapId = MapHandle.getMapIdForScenePath(nowMapPath)
        CacheContorl.nowMapId = upMapId
        seeMapFlow()
    elif downMapButton != 'Null' and yrn == downMapButton:
        playerPosition = CacheContorl.playObject['object']['0']['Position']
        downMapSceneId = MapHandle.getMapSceneIdForSceneId(CacheContorl.nowMapId,playerPosition)
        nowMapPath = MapHandle.getPathForMapId(CacheContorl.nowMapId)
        downMapPath = os.path.join(nowMapPath,downMapSceneId)
        downMapId = MapHandle.getMapIdForPath(downMapPath)
        CacheContorl.nowMapId = downMapId
        seeMapFlow()

