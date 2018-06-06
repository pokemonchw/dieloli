import Panel.SeeMapPanel as seemappanel
import core.flow as flow
import design.ObjectMove as objectmove
import design.MapHandle as maphandle

def seeMapFlow():
    inputS = []
    mapCmd = seemappanel.seeMapPanel()
    startId1 = len(mapCmd)
    inputS = inputS + mapCmd
    movePathCmdData = seemappanel.seeMovePathPanel()
    movePathCmd = movePathCmdData['inputS']
    movePathList = movePathCmdData['scenePathList']
    seeMapCmd = seemappanel.backScenePanel(startId1)
    inputS = inputS + seeMapCmd + movePathCmd
    yrn = flow.askfor_All(inputS)
    backButton = str(startId1)
    if yrn in mapCmd:
        objectmove.playerMove(yrn)
    elif yrn == backButton:
        import flow.InScene as inscene
        inscene.getInScene_func()
    elif yrn in movePathCmd:
        moveListId = movePathCmd.index(yrn)
        movePath = movePathList[moveListId]
        objectmove.playerMove(movePath)

