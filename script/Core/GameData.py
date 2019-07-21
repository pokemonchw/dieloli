# -*- coding: UTF-8 -*-
from script.Core.GamePathConfig import gamepath
from script.Core import JsonHandle,CacheContorl
from dijkstar import Graph,find_path
import os

_gamedata = {}

# 初始化游戏数据
def gamedata():
    return _gamedata

# 载入路径下所有json文件
def _loaddir(dataPath):
    _gamedata.update(loadDirNow(dataPath))

sceneData = {}
mapData = {}
def loadDirNow(dataPath):
    nowData = {}
    if os.listdir(dataPath):
        for i in os.listdir(dataPath):
            nowPath = os.path.join(dataPath,i)
            if os.path.isfile(nowPath):
                nowFile = i.split('.')
                if len(nowFile) > 1:
                    if nowFile[1] == 'json':
                        if nowFile[0] == 'Scene':
                            nowSceneData = {}
                            mapSystemPath = getMapSystemPathForPath(nowPath)
                            mapSystemPathStr = getMapSystemPathStr(mapSystemPath)
                            loadSceneData = JsonHandle._loadjson(nowPath)
                            nowSceneData.update(loadSceneData)
                            nowSceneData['SceneCharacterData'] = {}
                            nowSceneData['ScenePath'] = mapSystemPath
                            nowSceneData = {mapSystemPathStr:nowSceneData}
                            sceneData.update(nowSceneData)
                            nowSceneTag = loadSceneData['SceneTag']
                            if nowSceneTag not in CacheContorl.placeData:
                                CacheContorl.placeData[nowSceneTag] = []
                            CacheContorl.placeData[nowSceneTag].append(mapSystemPathStr)
                        elif nowFile[0] == 'Map':
                            nowMapData = {}
                            mapSystemPath = getMapSystemPathForPath(nowPath)
                            nowMapData['MapPath'] = mapSystemPath
                            with open(os.path.join(dataPath,"Map"), 'r') as nowReadFile:
                                drawData = nowReadFile.read()
                                nowMapData['MapDraw'] = getPrintMapData(drawData)
                            mapSystemPathStr = getMapSystemPathStr(mapSystemPath)
                            nowMapData.update(JsonHandle._loadjson(nowPath))
                            CacheContorl.nowInitMapId = mapSystemPathStr
                            sortedPathData = getSortedMapPathData(nowMapData['PathEdge'])
                            nowMapData['SortedPath'] = sortedPathData
                            mapData[mapSystemPathStr] = nowMapData
                        else:
                            nowData[nowFile[0]] = JsonHandle._loadjson(nowPath)
            else:
                nowData[i] = loadDirNow(nowPath)
    return nowData

# 获取地图下各节点到目标节点最短路径数据
def getSortedMapPathData(mapData):
    graph = Graph()
    sortedPathData = {}
    for node in mapData.keys():
        for target in mapData[node]:
            graph.add_edge(node,target,{'cost':mapData[node][target]})
    cost_func = lambda u,v,e,prev_e:e['cost']
    for node in mapData.keys():
        newData = {
            node:{}
        }
        for target in mapData.keys():
            if target != node:
                findPathData = find_path(graph,node,target,cost_func=cost_func)
                newData[node].update({target:{"Path":findPathData.nodes[1:],"Time":findPathData.costs}})
        sortedPathData.update(newData)
    return sortedPathData

# 从路径获取地图系统路径
def getMapSystemPathForPath(nowPath):
    currentDir = os.path.dirname(os.path.abspath(nowPath))
    currentDirStr = str(currentDir)
    mapStartList = currentDirStr.split('map')
    currentDirStr = mapStartList[1]
    mapSystemPath = currentDirStr.split(os.sep)
    mapSystemPath = mapSystemPath[1:]
    return mapSystemPath

# 获取地图系统路径字符串
def getMapSystemPathStr(nowPath):
    return os.sep.join(nowPath)

# 获取地图绘制数据
def getPrintMapData(mapDraw):
    mapYList = mapDraw.split('\n')
    newMapYList = []
    mapXListCmdData = {}
    mapXFixList = []
    mapXCmdIdData = {}
    for mapXListId in range(len(mapYList)):
        setMapButton = False
        mapXList = mapYList[mapXListId]
        mapXListCmdList = []
        cmdIdList = []
        newXList = ''
        nowCmd = ''
        nowCmdId = 0
        i = 0
        while i in range(len(mapXList)):
            if setMapButton == False and mapXList[i:i+11] != '<mapbutton>':
                newXList += mapXList[i]
            elif setMapButton == False and mapXList[i:i+11] == '<mapbutton>':
                i += 10
                setMapButton = True
            elif setMapButton == True and mapXList[i:i+12] != '</mapbutton>':
                nowCmd += mapXList[i]
            else:
                setMapButton = False
                mapXListCmdList.append(nowCmd)
                cmdIdList.append(len(newXList))
                nowCmd = ''
                nowCmdId = i + 12
                i += 11
            i += 1
        mapXListCmdData[mapXListId] = mapXListCmdList
        newMapYList.append(newXList)
        mapXCmdIdData[mapXListId] = cmdIdList
    return {"Draw":newMapYList,"Cmd":mapXListCmdData,"CmdId":mapXCmdIdData}

# 获取路径下所有子路径列表
def getPathList(pathData):
    pathList = []
    for i in os.listdir(pathData):
        path = os.path.join(pathData,i)
        if os.path.isdir(path):
            pathList.append(i)
    return pathList

# 游戏初始化
def init():
    datapath = os.path.join(gamepath,'data')
    _loaddir(datapath)
