import os
from core.pycfg import gamepath
import core.GameConfig as config
import core.RichText as richtext
import core.data as data
import core.EraPrint as eprint
import core.PyCmd as pycmd
import core.CacheContorl as cache

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
    if characterId in scenePlayerData[oldSceneId]:
        scenePlayerData[oldSceneId].remove(characterId)
    if characterId in scenePlayerData[newSceneId]:
        pass
    else:
        cache.playObject['object'][characterId]['Position'] = newSceneId
        scenePlayerData[newSceneId].append(characterId)
    cache.sceneData['ScenePlayerData'] = scenePlayerData
    pass

# 计算寻路路径
def getPathfinding(mapId,origin,destination):
    mapId = int(mapId)
    origin = str(origin)
    destination = str(destination)
    mapData = cache.mapData['MapData'][mapId]
    pathEdge = mapData['PathEdge']
    book = set()
    minv = origin
    INF = 999
    needTime = dict((k, INF) for k in pathEdge.keys())
    needTime[origin] = 0
    nodePath = dict((k, []) for k in pathEdge.keys())
    nodePath[origin] = [origin]
    timeList = nodePath.copy()
    timeList[origin] = [0]
    while len(book) < len(pathEdge):
        book.add(minv)
        for w in pathEdge[minv]:
            if needTime[minv] + pathEdge[minv][w] < needTime[w]:
                needTime[w] = needTime[minv] + pathEdge[minv][w]
                nodePath[w] = nodePath[minv].copy()
                nodePath[w].append(w)
                timeList[w] = timeList[minv].copy()
                timeList[w].append(pathEdge[minv][w])
        new = INF
        for v in needTime.keys():
            if v in book: continue
            if needTime[v] < new:
                new = needTime[v]
                minv = v
    return {"Path":nodePath[destination],"Time":timeList[destination]}

# 载入地图下对应场景数据
def getSceneDataForMap(mapId,mapSceneId):
    mapId = int(mapId)
    mapSceneId = str(mapSceneId)
    mapPath = cache.mapData['MapPathData'][mapId]
    scenePath = os.path.join(mapPath,mapSceneId)
    sceneData = getSceneDataForPath(scenePath)
    return sceneData

# 获取全局场景id对应的地图场景id
def getMapSceneIdForSceneId(sceneId):
    sceneId = int(sceneId)
    mapId = getMapIdForScene(sceneId)
    mapPath = cache.mapData['MapPathData'][mapId]
    scenePath = cache.sceneData['ScenePathData'][sceneId]
    sceneList = os.listdir(mapPath)
    sceneLoadList = []
    for scene in sceneList:
        sceneLoadPath = os.path.join(mapPath,scene)
        sceneLoadList.append(sceneLoadPath)
    mapSceneId = sceneLoadList.index(scenePath)
    return mapSceneId

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
    if 'Scene.jspn' in os.listdir(scenePath):
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
    scenePlayerData = {}
    for dirpath, dirnames, filenames in os.walk(mapDataDir):
        for i in range(0,len(filenames)):
            filename = filenames[i]
            if filename == 'Scene.json':
                scenePath = os.path.join(dirpath,filename)
                scene = data._loadjson(scenePath)
                sceneData.append(scene)
                scenePathData.append(dirpath)
                sceneId = str(i)
                scenePlayerData[sceneId] = []
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
        sceneId = str(i)
        scenePlayerData[sceneId] = []
    cache.sceneData['ScenePlayerData'] = scenePlayerData