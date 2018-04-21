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
    mapText = cache.mapData['MapData'][mapId]
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
def getMapForPath(mapPath):
    mapPath = os.path.abspath(os.path.join(mapPath, '..'))
    if 'Map' in os.listdir(mapPath):
        pass
    else:
        mapPath = getMapForPath(mapPath)
    return mapPath

# 获取地图下所有场景
def getSceneListForMap(mapId):
    mapId = int(mapId)
    mapPath = cache.mapData['MapPathData'][mapId]
    sceneList = data.getPathList(mapPath)
    return sceneList

# 获取地图上所有节点的xy坐标
def getMapCoordinate(mapId):
    mapText = cache.mapData['MapData'][mapId]
    sceneList = getSceneListForMap(mapId)
    inputCmd = ''
    passList = []
    mapYList = mapText.split('\n')
    cmdCoordinate = {}
    for y in range(0,len(mapYList)):
        mapXList = mapYList[y]
        mapXListStyle = richtext.setRichTextPrint(mapXList, 'standard')
        mapXList = richtext.removeRichCache(mapXList)
        for i in range(0, len(mapXList)):
            if str(i) not in passList:
                if mapXListStyle[i] == 'mapbutton':
                    inputCmd = inputCmd + mapXList[i]
                    for n in range(i + 1, len(mapXList)):
                        if mapXListStyle[n] == 'mapbutton':
                            inputCmd = inputCmd + mapXList[n]
                            passList.append(str(n))
                        else:
                            break
                    if inputCmd in sceneList:
                        cmdXY = {'x':i,'y':y}
                        cmdCoordinate[inputCmd] = cmdXY
                    inputCmd = ''
    return cmdCoordinate

# 载入所有场景数据
def initSceneData():
    sceneData = []
    scenePathData = []
    for dirpath, dirnames, filenames in os.walk(mapDataDir):
        for filename in filenames:
            if filename == 'Scene.json':
                scenePath = os.path.join(dirpath,filename)
                scene = data._loadjson(scenePath)
                sceneData.append(scene)
                scenePathData.append(dirpath)
    cache.sceneData = {"SceneData":sceneData,"ScenePathData":scenePathData}

# 载入所有地图数据
def initMapData():
    mapData = []
    mapPathData = []
    for dirpath, dirnames, filenames in os.walk(mapDataDir):
        for filename in filenames:
            if filename == 'Map':
                mapPath = os.path.join(dirpath,filename)
                openMap = open(mapPath)
                mapText = openMap.read()
                mapData.append(mapText)
                mapPathData.append(dirpath)
    cache.mapData = {"MapData":mapData,"MapPathData":mapPathData}