from script.Core import CacheContorl,TextLoading,EraPrint,ValueHandle
from script.Design import MapHandle,CmdButtonQueue

# 用于绘制地图的面板
def seeMapPanel():
    inputS = []
    titleText = TextLoading.getTextData(TextLoading.stageWordId, '78')
    EraPrint.plt(titleText)
    sceneId = CacheContorl.playObject['object']['0']['Position']
    mapId = MapHandle.getMapIdForScene(sceneId)
    inputS = inputS + MapHandle.printMap(mapId)
    return inputS

# 用于绘制移动路径按钮的面板
def seeMovePathPanel():
    inputS = []
    sceneId = CacheContorl.playObject['object']['0']['Position']
    mapId = MapHandle.getMapIdForScene(sceneId)
    mapData = CacheContorl.mapData['MapData'][mapId]
    movePathInfo = TextLoading.getTextData(TextLoading.messageId,'27')
    EraPrint.p(movePathInfo)
    EraPrint.p('\n')
    pathEdge = mapData['PathEdge']
    mapSceneId = str(MapHandle.getMapSceneIdForSceneId(mapId, sceneId))
    scenePath = pathEdge[mapSceneId]
    scenePathList = ValueHandle.dictKeysToList(scenePath)
    try:
        scenePathList.remove(mapSceneId)
    except ValueError:
        pass
    if len(scenePathList) > 0:
        sceneCmd = []
        for scene in scenePathList:
            loadSceneData = MapHandle.getSceneDataForMap(mapId, scene)
            sceneName = loadSceneData['SceneName']
            sceneCmd.append(sceneName)
        yrn = CmdButtonQueue.optionstr(cmdList=None, cmdListData=sceneCmd, cmdColumn=4, askfor=False, cmdSize='center')
        inputS = inputS + yrn
    else:
        errorMoveText = TextLoading.getTextData(TextLoading.messageId, '28')
        EraPrint.p(errorMoveText)
    EraPrint.pline()
    return {'inputS':inputS,'scenePathList':scenePathList}

# 用于绘制通常按钮面板
def backScenePanel(startId):
    inputS = []
    mapCmdList = CmdButtonQueue.optionint(CmdButtonQueue.seemap, askfor=False, startId=startId)
    inputS = inputS + mapCmdList
    return inputS
