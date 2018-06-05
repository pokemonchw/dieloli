import os
from core.pycfg import gamepath
import core.GameConfig as config
import core.RichText as richtext
import core.data as data
import core.EraPrint as eprint
import core.PyCmd as pycmd
import core.CacheContorl as cache
import core.TextHandle as texthandle
import core.ValueHandle as valuehandle

language = config.language
mapDataDir = os.path.join(gamepath, 'data',language, 'map')

# 输出地图
def printMap(mapId):
    mapText = cache.mapData['MapTextData'][mapId]
    sceneList = getSceneListForMap(mapId)
    inputS = []
    inputCmd = ''
    passList = []
    mapYList = mapText.split('\n')
    for mapXList in mapYList:
        mapXListStyle = richtext.setRichTextPrint(mapXList,'standard')
        mapXList = richtext.removeRichCache(mapXList)
        mapXFix = texthandle.align(mapXList,'center',True)
        eprint.p(mapXFix)
        for i in range(0, len(mapXList)):
            if str(i) not in passList:
                if mapXListStyle[i] == 'mapbutton':
                    inputCmd = inputCmd + mapXList[i]
                    for n in range(i + 1,len(mapXList)):
                        if mapXListStyle[n] == 'mapbutton':
                            inputCmd = inputCmd + mapXList[n]
                            passList.append(str(n))
                        else:
                            break
                    if inputCmd in sceneList:
                        pycmd.pcmd(inputCmd, inputCmd, None)
                        inputS.append(inputCmd)
                    else:
                        eprint.p(inputCmd,'standard')
                    inputCmd = ''
                else:
                    eprint.p(mapXList[i], mapXListStyle[i])
        eprint.p('\n')
    return inputS

# 获取场景所在地图
def getMapForScene(sceneId):
    sceneId = int(sceneId)
    scenePath = cache.sceneData['ScenePathData'][sceneId]
    mapPath = getMapForPath(scenePath)
    return mapPath

# 查找场景所在地图
def getMapForPath(scenePath):
    mapPath = os.path.abspath(os.path.join(scenePath, '..'))
    if 'Map' in os.listdir(mapPath):
        pass
    else:
        mapPath = getMapForPath(mapPath)
    return mapPath

# 查找场景所在地图ID
def getMapIdForScene(sceneId):
    mapPath = getMapForScene(sceneId)
    mapData = cache.mapData['MapPathData']
    mapId = mapData.index(mapPath)
    return mapId

# 获取地图下所有场景
def getSceneListForMap(mapId):
    mapId = int(mapId)
    mapPath = cache.mapData['MapPathData'][mapId]
    sceneList = data.getPathList(mapPath)
    return sceneList

# 场景移动
def playerMoveScene(oldSceneId,newSceneId,characterId):
    scenePlayerData = cache.sceneData['ScenePlayerData']
    characterId = str(characterId)
    oldSceneId = int(oldSceneId)
    newSceneId = int(newSceneId)
    if characterId in scenePlayerData[oldSceneId]:
        scenePlayerData[oldSceneId].remove(characterId)
    if characterId in scenePlayerData[newSceneId]:
        pass
    else:
        cache.playObject['object'][characterId]['Position'] = newSceneId
        scenePlayerData[newSceneId].append(characterId)
    cache.sceneData['ScenePlayerData'] = scenePlayerData

# 计算寻路路径
def getPathfinding(mapId,nowNode,targetNode,pathNodeList = [],pathTimeList = [],pathTime = 0,pathList = [],timeList = []):
    mapId = int(mapId)
    nowNode = str(nowNode)
    targetNode = str(targetNode)
    mapData = cache.mapData['MapData'][mapId]
    pathEdge = mapData['PathEdge']
    targetListDict = pathEdge[nowNode]
    targetList = valuehandle.dictKeysToList(targetListDict)
    if nowNode == targetNode:
        return 'End'
    else:
        for i in range(0,len(targetList)):
            target = targetList[i]
            if target in pathNodeList:
                pass
            else:
                targetTime = targetListDict[target]
                pathTime = pathTime + targetTime
                findPath = pathNodeList.copy()
                if findPath == []:
                    findPath = [nowNode]
                    findTime = [0]
                else:
                    findTime = pathTimeList.copy()
                findPath.append(target)
                findTime.append(targetTime)
                if target == targetNode:
                    pathList.append(findPath)
                    timeList.append(findTime)
                else:
                    pathEdgeNow = pathEdge[target].copy()
                    pathEdgeNow.pop(nowNode)
                    targetNodeInTargetList = pathEdgeNow
                    targetNodeInTargetToList = valuehandle.dictKeysToList(targetNodeInTargetList)
                    for i in range(0,len(targetNodeInTargetToList)):
                        targetNodeInTarget = targetNodeInTargetToList[i]
                        findPath.append(targetNodeInTarget)
                        findTime.append(targetNodeInTargetList[targetNodeInTarget])
                        pathData = getPathfinding(mapId,targetNodeInTarget,targetNode,findPath,findTime,pathTime,pathList,timeList)
                        if pathData == 'Null':
                            pass
                        elif pathData == 'End':
                            pass
                        else:
                            pathList.append(pathData['Path'])
                            timeList.append(pathData['Time'])
        if len(pathList) > 0:
            pathId = 0
            needTime = 'Null'
            for i in range(0,len(timeList)):
                nowTime = 0
                for index in range(0,len(timeList[i])):
                    if index == 0:
                        needTime = 0
                    nowTime = needTime + timeList[i][index]
                if needTime == 'Null' or nowTime < needTime:
                    needTime = nowTime
                    pathId = i
            pathData = {'Path':pathList[pathId],'Time':timeList[pathId]}
            return pathData
        else:
            return 'Null'

# 载入地图下对应场景数据
def getSceneDataForMap(mapId,mapSceneId):
    mapId = int(mapId)
    mapSceneId = str(mapSceneId)
    mapPath = cache.mapData['MapPathData'][mapId]
    scenePath = os.path.join(mapPath,mapSceneId)
    sceneData = getSceneDataForPath(scenePath)
    return sceneData

# 获取全局场景id对应的地图场景id
def getMapSceneIdForSceneId(mapId,sceneId):
    sceneId = int(sceneId)
    scenePath = cache.sceneData['ScenePathData'][sceneId]
    mapId = int(mapId)
    sceneInPath = getMapScenePathForScenePath(mapId,scenePath)
    mapPath = cache.mapData['MapPathData'][mapId]
    mapSceneId = judgeSonMapInMap(mapPath, sceneInPath)
    return mapSceneId

# 获取从场景路径获取对应地图下路径
def getMapScenePathForScenePath(mapId,scenePath):
    mapPath = cache.mapData['MapPathData'][mapId]
    sceneInPath = os.path.abspath(os.path.join(scenePath, '..'))
    if mapPath == sceneInPath:
        nowPath = scenePath
    else:
        nowPath = getMapScenePathForScenePath(mapId,sceneInPath)
    return nowPath


# 获取地图场景id对应的全剧场景id
def getSceneIdForMapSceneId(mapId,mapSceneId):
    scenePath = getScenePathForMapSceneId(mapId,mapSceneId)
    sceneData = cache.sceneData.copy()
    scenePathData = sceneData['ScenePathData']
    sceneId = scenePathData.index(scenePath)
    return sceneId

# 判断地图在指定地图中的位置
def judgeSonMapInMap(mapPath,sonMapPath):
    mapDirList = os.listdir(mapPath)
    mapPathList = []
    for i in mapDirList:
        loadPath = os.path.join(mapPath,i)
        if os.path.isfile(loadPath):
            pass
        else:
            mapPathList.append(loadPath)
    if sonMapPath in mapPathList:
        sonMapId = mapPathList.index(sonMapPath)
    else:
        loadSonMapPath = os.path.abspath(os.path.join(sonMapPath, '..'))
        sonMapId = judgeSonMapInMap(mapPath,loadSonMapPath)
    return sonMapId

# 从对应地图场景id查找场景路径
def getScenePathForMapSceneId(mapId,mapSceneId):
    mapId = int(mapId)
    mapSceneId = str(mapSceneId)
    mapPath = cache.mapData['MapPathData'][mapId]
    scenePath = os.path.join(mapPath,mapSceneId)
    scenePath = getScenePathForTrue(scenePath)
    return scenePath

# 获取有效场景路径
def getScenePathForTrue(scenePath):
    if 'Scene.json' in os.listdir(scenePath):
        pass
    else:
        scenePath = os.path.join(scenePath,'0')
        scenePath = getScenePathForTrue(scenePath)
    return scenePath

# 从对应路径查找场景数据
def getSceneDataForPath(scenePath):
    if 'Scene.json' in os.listdir(scenePath):
        scenePath = os.path.join(scenePath,'Scene.json')
        sceneData = data._loadjson(scenePath)
    else:
        scenePath = os.path.join(scenePath,'0')
        sceneData = getSceneDataForPath(scenePath)
    return sceneData

# 载入所有场景数据
def initSceneData():
    sceneData = []
    scenePathData = []
    scenePlayerData = []
    for dirpath, dirnames, filenames in os.walk(mapDataDir):
        for i in range(0,len(filenames)):
            filename = filenames[i]
            if filename == 'Scene.json':
                scenePath = os.path.join(dirpath,filename)
                scene = data._loadjson(scenePath)
                sceneData.append(scene)
                scenePathData.append(dirpath)
                scenePlayerData.append([])
    cache.sceneData = {"SceneData":sceneData,"ScenePathData":scenePathData,"ScenePlayerData":scenePlayerData}

# 载入所有地图数据
def initMapData():
    mapData = []
    mapPathData = []
    mapTextData = []
    for dirpath, dirnames, filenames in os.walk(mapDataDir):
        for filename in filenames:
            if filename == 'Map':
                mapPath = os.path.join(dirpath,'Map')
                mapDataPath = os.path.join(dirpath,'Map.json')
                openMap = open(mapPath)
                mapText = openMap.read()
                mapJsonData = data._loadjson(mapDataPath)
                mapData.append(mapJsonData)
                mapTextData.append(mapText)
                mapPathData.append(dirpath)
    cache.mapData = {"MapData":mapData,"MapPathData":mapPathData,"MapTextData":mapTextData}

# 初始化场景上的角色
def initScanePlayerData():
    scenePlayerData = cache.sceneData['ScenePlayerData']
    for i in range(0,len(scenePlayerData)):
        scenePlayerData[i] = []
    cache.sceneData['ScenePlayerData'] = scenePlayerData

# 获取场景上所有角色的数据
def getScenePlayerData(sceneId):
    playerData = cache.playObject['object']
    scenePlayerData = []
    scenePlayerDataList = cache.sceneData['ScenePlayerData'][sceneId]
    for i in scenePlayerDataList:
        scenePlayerData.append(playerData[i])
    return scenePlayerData

# 获取场景上所有角色的姓名列表
def getScenePlayerNameList(sceneId):
    scenePlayerData = getScenePlayerData(sceneId)
    scenePlayerNameList = []
    for i in scenePlayerData:
        scenePlayerNameList.append(i['Name'])
    return scenePlayerNameList

# 获取场景上所有角色的id列表
def getScenePlayerIdList(sceneId):
    scenePlayerDataList = cache.sceneData['ScenePlayerData'][sceneId]
    scenePlayerIdList = []
    for i in scenePlayerDataList:
        scenePlayerIdList.append(i)
    return scenePlayerIdList
