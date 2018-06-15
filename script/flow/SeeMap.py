from core import FlowHandle
from design import ObjectMove
from Panel import SeeMapPanel

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
    if yrn in mapCmd:
        ObjectMove.playerMove(yrn)
    elif yrn == backButton:
        import flow.InScene as inscene
        inscene.getInScene_func()
    elif yrn in movePathCmd:
        moveListId = movePathCmd.index(yrn)
        movePath = movePathList[moveListId]
        ObjectMove.playerMove(movePath)

