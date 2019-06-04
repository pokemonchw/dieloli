from script.Core import FlowHandle,CacheContorl,PyCmd
from script.Design import CharacterMove,MapHandle,PanelStateHandle
from script.Panel import SeeMapPanel

def seeMapFlow():
    while(True):
        PyCmd.clr_cmd()
        inputS = []
        mapCmd = SeeMapPanel.seeMapPanel()
        startId1 = len(mapCmd)
        inputS = inputS + mapCmd
        movePathCmdData = SeeMapPanel.seeMovePathPanel()
        movePathCmd = movePathCmdData['inputS']
        movePathList = movePathCmdData['scenePathList']
        showSceneNameListCmd = SeeMapPanel.showSceneNameListPanel()
        seeMapCmd = SeeMapPanel.backScenePanel(startId1)
        inputS = inputS + seeMapCmd + movePathCmd + [showSceneNameListCmd]
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
        nowMap = CacheContorl.nowMap.copy()
        if yrn in mapCmd:
            nowTargetPath = MapHandle.getScenePathForMapSceneId(nowMap,yrn)
            CharacterMove.ownCharcterMove(nowTargetPath)
            break
        elif yrn == backButton:
            CacheContorl.nowMap = []
            CacheContorl.nowFlowId = 'in_scene'
            break
        elif yrn in movePathCmd:
            moveListId = movePathCmd.index(yrn)
            moveId = movePathList[moveListId]
            nowTargetPath = MapHandle.getScenePathForMapSceneId(nowMap,moveId)
            CharacterMove.ownCharcterMove(nowTargetPath)
            break
        elif upMapButton != 'Null' and yrn == upMapButton:
            upMapPath = MapHandle.getMapForPath(nowMap)
            CacheContorl.nowMap = upMapPath
        elif downMapButton != 'Null' and yrn == downMapButton:
            characterPosition = CacheContorl.characterData['character']['0']['Position']
            downMapSceneId = MapHandle.getMapSceneIdForScenePath(CacheContorl.nowMap,characterPosition)
            nowMap.append(downMapSceneId)
            CacheContorl.nowMap = nowMap
        elif yrn == showSceneNameListCmd:
            PanelStateHandle.panelStateChange(showSceneNameListCmd)
