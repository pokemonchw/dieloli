import Panel.SeeMapPanel as seemappanel
import core.flow as flow
import design.ObjectMove as objectmove

def seeMapFlow():
    inputS = []
    mapCmd = seemappanel.seeMapPanel()
    startId1 = len(mapCmd)
    inputS = inputS + mapCmd
    movePathCmd = seemappanel.seeMovePathPanel()
    inputS = inputS + movePathCmd
    seeMapCmd = seemappanel.backScenePanel(startId1)
    inputS = inputS + seeMapCmd
    yrn = flow.askfor_All(inputS)
    if yrn in mapCmd:
        objectmove.playerMove(yrn)
    pass

