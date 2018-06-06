import core.CacheContorl as cache
import design.MapHandle as maphandle
import core.TextLoading as textload
import core.EraPrint as eprint
import core.ValueHandle as valuehandle
import design.Ans as ans

# 用于绘制地图的面板
def seeMapPanel():
    inputS = []
    titleText = textload.getTextData(textload.stageWordId, '78')
    eprint.plt(titleText)
    sceneId = cache.playObject['object']['0']['Position']
    mapId = maphandle.getMapIdForScene(sceneId)
    inputS = inputS + maphandle.printMap(mapId)
    return inputS

# 用于绘制移动路径按钮的面板
def seeMovePathPanel():
    inputS = []
    sceneId = cache.playObject['object']['0']['Position']
    mapId = maphandle.getMapIdForScene(sceneId)
    mapData = cache.mapData['MapData'][mapId]
    movePathInfo = textload.getTextData(textload.messageId,'27')
    eprint.p(movePathInfo)
    eprint.p('\n')
    pathEdge = mapData['PathEdge']
    mapSceneId = str(maphandle.getMapSceneIdForSceneId(mapId, sceneId))
    scenePath = pathEdge[mapSceneId]
    scenePathList = valuehandle.dictKeysToList(scenePath)
    try:
        scenePathList.remove(mapSceneId)
    except ValueError:
        pass
    if len(scenePathList) > 0:
        sceneCmd = []
        for scene in scenePathList:
            loadSceneData = maphandle.getSceneDataForMap(mapId, scene)
            sceneName = loadSceneData['SceneName']
            sceneCmd.append(sceneName)
        yrn = ans.optionstr(cmdList=None, cmdListData=sceneCmd, cmdColumn=4, askfor=False, cmdSize='center')
        inputS = inputS + yrn
    else:
        errorMoveText = textload.getTextData(textload.messageId, '28')
        eprint.p(errorMoveText)
    eprint.pline()
    return {'inputS':inputS,'scenePathList':scenePathList}

# 用于绘制通常按钮面板
def backScenePanel(startId):
    inputS = []
    mapCmdList = ans.optionint(ans.seemap, askfor=False, startId=startId)
    inputS = inputS + mapCmdList
    return inputS
