from script.Core import FlowHandle,CacheContorl
from script.Design import ObjectMove,MapHandle
from script.Panel import SeeMapPanel

def seeMapFlow():
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
    mapId = CacheContorl.nowMapId
    if yrn in mapCmd:
        nowTargetPath = MapHandle.getScenePathForMapSceneId(mapId,yrn)
        ObjectMove.playerMove(nowTargetPath)
    elif yrn == backButton:
        import Flow.InScene as inscene
        inscene.getInScene_func()
    elif yrn in movePathCmd:
        moveListId = movePathCmd.index(yrn)
        moveId = movePathList[moveListId]
        nowTargetPath = MapHandle.getScenePathForMapSceneId(mapId,moveId)
        ObjectMove.playerMove(nowTargetPath)

