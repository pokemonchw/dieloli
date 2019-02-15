from script.Core import CacheContorl,TextLoading,EraPrint,ValueHandle
from script.Design import MapHandle,CmdButtonQueue

# 用于绘制地图的面板
def seeMapPanel():
    inputS = []
    titleText = TextLoading.getTextData(TextLoading.stageWordPath, '78')
    EraPrint.plt(titleText)
    mapId = CacheContorl.nowMapId
    inputS = inputS + MapHandle.printMap(mapId)
    return inputS

# 用于绘制移动路径按钮的面板
def seeMovePathPanel():
    inputS = []
    sceneId = CacheContorl.playObject['object']['0']['Position']
    mapId = CacheContorl.nowMapId
    mapData = CacheContorl.mapData['MapData'][mapId]
    movePathInfo = TextLoading.getTextData(TextLoading.messagePath,'27')
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
        errorMoveText = TextLoading.getTextData(TextLoading.messagePath, '28')
        EraPrint.p(errorMoveText)
    EraPrint.pline()
    return {'inputS':inputS,'scenePathList':scenePathList}

# 用于绘制通常按钮面板
def backScenePanel(startId):
    seeMapCmd = []
    nowPosition = CacheContorl.playObject['object']['0']['Position']
    nowPositionMapId = MapHandle.getMapIdForScene(nowPosition)
    cmdData = TextLoading.getTextData(TextLoading.cmdPath,CmdButtonQueue.seemap)
    seeMapCmd.append(cmdData[0])
    if str(nowPositionMapId) != '0' and str(CacheContorl.nowMapId) != '0':
        seeMapCmd.append(cmdData[1])
    if str(nowPositionMapId) != str(CacheContorl.nowMapId):
        seeMapCmd.append(cmdData[2])
    mapCmdList = CmdButtonQueue.optionint(cmdList=None,cmdListData=seeMapCmd,cmdColumn=3,askfor=False,cmdSize='center',startId=startId)
    return mapCmdList
