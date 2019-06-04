from script.Core import CacheContorl,TextLoading,EraPrint,PyCmd
from script.Design import MapHandle,CmdButtonQueue

panelStateTextData = TextLoading.getTextData(TextLoading.cmdPath,'cmdSwitch')
panelStateOnText = panelStateTextData[1]
panelStateOffText = panelStateTextData[0]

# 用于绘制地图的面板
def seeMapPanel():
    inputS = []
    titleText = TextLoading.getTextData(TextLoading.stageWordPath, '78')
    nowMap = CacheContorl.nowMap
    nowMapMapSystemStr = MapHandle.getMapSystemPathStrForList(nowMap)
    mapName = CacheContorl.mapData[nowMapMapSystemStr]['MapName']
    EraPrint.plt(titleText + ': ' + mapName + ' ')
    inputS = inputS + MapHandle.printMap(nowMap)
    return inputS

# 用于绘制移动路径按钮的面板
def seeMovePathPanel():
    inputS = []
    nowScene = CacheContorl.characterData['character']['0']['Position']
    nowMap = CacheContorl.nowMap
    nowMapStr = MapHandle.getMapSystemPathStrForList(nowMap)
    mapData = CacheContorl.mapData[nowMapStr]
    movePathInfo = TextLoading.getTextData(TextLoading.messagePath,'27')
    EraPrint.p(movePathInfo)
    EraPrint.p('\n')
    pathEdge = mapData['PathEdge']
    mapSceneId = str(MapHandle.getMapSceneIdForScenePath(nowMap, nowScene))
    scenePath = pathEdge[mapSceneId]
    scenePathList = list(scenePath.keys())
    try:
        scenePathList.remove(mapSceneId)
    except ValueError:
        pass
    if len(scenePathList) > 0:
        sceneCmd = []
        for scene in scenePathList:
            loadSceneData = MapHandle.getSceneDataForMap(nowMap, scene)
            sceneName = loadSceneData['SceneName']
            sceneCmd.append(sceneName)
        yrn = CmdButtonQueue.optionstr(cmdList=None, cmdListData=sceneCmd, cmdColumn=4, askfor=False, cmdSize='center')
        inputS = inputS + yrn
    else:
        errorMoveText = TextLoading.getTextData(TextLoading.messagePath, '28')
        EraPrint.p(errorMoveText)
    EraPrint.pline()
    return {'inputS':inputS,'scenePathList':scenePathList}

# 用于绘制场景名字列表面板
def showSceneNameListPanel():
    titleText = TextLoading.getTextData(TextLoading.stageWordPath,'86')
    EraPrint.p(titleText)
    panelState = CacheContorl.panelState['SeeSceneNameListPanel']
    if panelState == '0':
        PyCmd.pcmd(panelStateOffText,"SeeSceneNameListPanel")
        EraPrint.p('\n')
        nowMap = CacheContorl.nowMap
        nowMapMapSystemStr = MapHandle.getMapSystemPathStrForList(nowMap)
        sceneNameData = MapHandle.getSceneNameListForMapPath(nowMapMapSystemStr)
        sceneNameList = []
        for scene in sceneNameData:
            sceneNameList.append(scene + ':' + sceneNameData[scene])
        EraPrint.plist(sceneNameList,5,'center')
    else:
        PyCmd.pcmd(panelStateOnText,'SeeSceneNameListPanel')
        EraPrint.p('\n')
    EraPrint.plittleline()
    return 'SeeSceneNameListPanel'

# 用于绘制通常按钮面板
def backScenePanel(startId):
    seeMapCmd = []
    nowPosition = CacheContorl.characterData['character']['0']['Position']
    nowMap = MapHandle.getMapForPath(nowPosition)
    cmdData = TextLoading.getTextData(TextLoading.cmdPath,CmdButtonQueue.seemap)
    seeMapCmd.append(cmdData[0])
    if nowMap != [] and CacheContorl.nowMap != []:
        seeMapCmd.append(cmdData[1])
    if nowMap != CacheContorl.nowMap:
        seeMapCmd.append(cmdData[2])
    mapCmdList = CmdButtonQueue.optionint(cmdList=None,cmdListData=seeMapCmd,cmdColumn=3,askfor=False,cmdSize='center',startId=startId)
    return mapCmdList
