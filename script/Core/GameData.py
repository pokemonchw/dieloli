# -*- coding: UTF-8 -*-
from script.Core.GamePathConfig import gamepath
from script.Core import JsonHandle,CacheContorl
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
                            nowSceneData.update(JsonHandle._loadjson(nowPath))
                            nowSceneData['SceneCharacterData'] = []
                            nowSceneData['ScenePath'] = mapSystemPath
                            nowSceneData = {mapSystemPathStr:nowSceneData}
                            sceneData.update(nowSceneData)
                        elif nowFile[0] == 'Map':
                            nowMapData = {}
                            mapSystemPath = getMapSystemPathForPath(nowPath)
                            nowMapData['MapPath'] = mapSystemPath
                            with open(os.path.join(dataPath,"Map"), 'r') as nowReadFile:
                                nowMapData['MapDraw'] = nowReadFile.read()
                            mapSystemPathStr = getMapSystemPathStr(mapSystemPath)
                            nowMapData.update(JsonHandle._loadjson(nowPath))
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
    sortedPathData = {}
    for node in mapData.keys():
        newData = {
            node:{}
        }
        for target in mapData.keys():
            if target != node:
                newData[node].update({target:getSortedPath(mapData,node,target)})
        sortedPathData.update(newData)
    return sortedPathData

# 计算寻路路径
def getSortedPath(pathEdge,nowNode,targetNode,pathNodeList=[],pathTimeList=[]):
    pathList = CacheContorl.pathList
    timeList = CacheContorl.pathTimeList
    nowNode = str(nowNode)
    targetNode = str(targetNode)
    targetListDict = pathEdge[nowNode].copy()
    targetList = list(targetListDict.keys())
    if nowNode == targetNode:
        return 'End'
    else:
        for i in range(0,len(targetList)):
            target = targetList[i]
            if target not in pathNodeList:
                targetTime = targetListDict[target]
                findPath = pathNodeList.copy()
                if findPath == []:
                    findPath = [nowNode]
                    findTime = [-1]
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
                    targetNodeInTargetList = pathEdgeNow.copy()
                    targetNodeInTargetToList = list(targetNodeInTargetList.keys())
                    for i in range(0,len(targetNodeInTargetToList)):
                        targetNodeInTarget = targetNodeInTargetToList[i]
                        findPath.append(targetNodeInTarget)
                        findTime.append(targetNodeInTargetList[targetNodeInTarget])
                        pathData = getSortedPath(pathEdge,targetNodeInTarget,targetNode,findPath,findTime)
                        if pathData == 'Null':
                            pass
                        elif pathData == 'End':
                            pathList.append(findPath)
                            timeList.append(findTime)
                        else:
                            pathList.append(pathData['Path'])
                            timeList.append(pathData['Time'])
        CacheContorl.pathTimeList = []
        CacheContorl.pathList = []
        return getMinimumPath(pathList,timeList)

# 获取最短路径
def getMinimumPath(pathList,timeList):
    if len(pathList) > 0:
        needTimeList = []
        for i in range(0,len(timeList)):
            needTimeList.append(getNeedTime(timeList[i]))
        pathId = needTimeList.index(min(needTimeList))
        return {"Path":pathList[pathId],"Time":timeList[pathId]}
    return 'Null'

# 获取路径所需时间
def getNeedTime(timeGroup):
    needTime = 0
    for i in timeGroup:
        needTime = needTime + i
    return needTime

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
