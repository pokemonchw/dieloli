from script.Core import CacheContorl,TextLoading,EraPrint,PyCmd
from script.Design import MapHandle,CmdButtonQueue

panelStateTextData = TextLoading.getTextData(TextLoading.cmdPath,'cmdSwitch')
panelStateOnText = panelStateTextData[1]
panelStateOffText = panelStateTextData[0]

def seeMapPanel() -> list:
    '''
    地图绘制面板
    '''
    inputS = []
    titleText = TextLoading.getTextData(TextLoading.stageWordPath, '78')
    nowMap = CacheContorl.nowMap
    nowMapMapSystemStr = MapHandle.getMapSystemPathStrForList(nowMap)
    mapName = CacheContorl.mapData[nowMapMapSystemStr]['MapName']
    EraPrint.plt(titleText + ': ' + mapName + ' ')
    inputS = inputS + MapHandle.printMap(nowMap)
    return inputS

def seeMovePathPanel() -> dict:
    '''
    当前场景可直接通往的移动路径绘制面板
    '''
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
    if mapSceneId in scenePathList:
        remove(mapSceneId)
    if len(scenePathList) > 0:
        sceneCmd = []
        for scene in scenePathList:
            nowMapStr = MapHandle.getMapSystemPathStrForList(nowMap)
            loadSceneData = MapHandle.getSceneDataForMap(nowMapStr, scene)
            sceneName = loadSceneData['SceneName']
            sceneCmd.append(sceneName)
        yrn = CmdButtonQueue.optionstr(cmdList=None, cmdListData=sceneCmd, cmdColumn=4, askfor=False, cmdSize='center')
        inputS = inputS + yrn
    else:
        errorMoveText = TextLoading.getTextData(TextLoading.messagePath, '28')
        EraPrint.p(errorMoveText)
    EraPrint.pline()
    return {'inputS':inputS,'scenePathList':scenePathList}

def showSceneNameListPanel() -> str:
    '''
    地图下场景名称绘制面板
    '''
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

def backScenePanel(startId:str) -> list:
    '''
    查看场景页面基础命令绘制面板
    Keyword arguments:
    startId -- 面板命令起始id
    '''
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
