from script.Core import FlowHandle,CacheContorl,PyCmd
from script.Design import CharacterMove,MapHandle
from script.Panel import SeeMapPanel

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
    nowPosition = CacheContorl.characterData['character']['0']['Position']
    nowPositionMap = MapHandle.getMapForPath(nowPosition)
    upMapButton = 'Null'
    downMapButton = 'Null'
    if nowPositionMap != [] and CacheContorl.nowMap != []:
        upMapButton = str(int(startId1) + 1)
    if nowPositionMap != CacheContorl.nowMap:
        if upMapButton == 'Null':
            downMapButton = str(int(startId1) + 1)
        else:
            downMapButton = str(int(startId1) + 2)
    nowMap = CacheContorl.nowMap
    if yrn in mapCmd:
        nowTargetPath = MapHandle.getScenePathForMapSceneId(nowMap,yrn)
        CharacterMove.ownCharcterMove(nowTargetPath)
    elif yrn == backButton:
        CacheContorl.nowMap = []
        import script.Flow.InScene as inscene
        inscene.getInScene_func()
    elif yrn in movePathCmd:
        moveListId = movePathCmd.index(yrn)
        moveId = movePathList[moveListId]
        nowTargetPath = MapHandle.getScenePathForMapSceneId(nowMap,moveId)
        CharacterMove.ownCharcterMove(nowTargetPath)
    elif upMapButton != 'Null' and yrn == upMapButton:
        upMapPath = MapHandle.getMapForPath(nowMap)
        CacheContorl.nowMap = upMapPath
        seeMapFlow()
    elif downMapButton != 'Null' and yrn == downMapButton:
        characterPosition = CacheContorl.characterData['character']['0']['Position']
        downMapSceneId = MapHandle.getMapSceneIdForScenePath(CacheContorl.nowMap,characterPosition)
        downMapPath = nowMap.append(downMapSceneId)
        CacheContorl.nowMap = downMapPath
        seeMapFlow()

