# -*- coding: UTF-8 -*-
from script.Core.GamePathConfig import gamepath
from script.Core import JsonHandle,CacheContorl
from dijkstar import Graph,find_path
import os

_gamedata = {}

def gamedata() -> dict:
    '''
    获取游戏数据
    '''
    return _gamedata

def _loaddir(dataPath:str):
    '''
    将路径下的游戏数据传入_gamedata
    Keyword arguments:
    datapath -- 要载入数据的路径
    '''
    _gamedata.update(loadDirNow(dataPath))

sceneData = {}
mapData = {}
def loadDirNow(dataPath:str):
    '''
    获取路径下的游戏数据
    Keyword arguments:
    dataPath -- 要载入数据的路径
    '''
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
                            if nowFile[0] == 'Equipment':
                                initClothingData(nowData[nowFile[0]]['Clothing'])
                            elif nowFile[0] == 'StatureDescription':
                                initStatureDescription(nowData[nowFile[0]]['Priority'])
            else:
                nowData[i] = loadDirNow(nowPath)
    return nowData

def initStatureDescription(sdData):
    '''
    初始化身材描述文本权重数据
    Keyword arguments:
    sdData -- 身材描述文本数据
    '''
    CacheContorl.statureDescritionPrioritionData = {priority:{i:len(sdData[priority][i]['Condition']) for i in range(len(sdData[priority]))} for priority in range(len(sdData))}

def getSortedMapPathData(mapData:dict) -> dict:
    '''
    获取地图下各节点到目标节点的最短路径数据
    Keyword arguments:
    mapData -- 地图节点数据
    '''
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

def getMapSystemPathForPath(nowPath:str) -> list:
    '''
    从地图文件路径获取游戏地图系统路径
    Keyword arguments:
    nowPath -- 地图文件路径
    '''
    currentDir = os.path.dirname(os.path.abspath(nowPath))
    currentDirStr = str(currentDir)
    mapStartList = currentDirStr.split('map')
    currentDirStr = mapStartList[1]
    mapSystemPath = currentDirStr.split(os.sep)
    mapSystemPath = mapSystemPath[1:]
    return mapSystemPath

def getMapSystemPathStr(nowPath:list) -> str:
    '''
    将游戏地图系统路径转换为字符串
    '''
    return os.sep.join(nowPath)

def getPrintMapData(mapDraw:str) -> dict:
    '''
    获取绘制地图的富文本和按钮数据
    Keyword arguments:
    mapDraw -- 绘制地图的原始数据
    '''
    mapYList = mapDraw.split('\n')
    newMapYList = []
    mapXListCmdData = {}
    mapXCmdIdData = {}
    for mapXListId in range(len(mapYList)):
        setMapButton = False
        mapXList = mapYList[mapXListId]
        mapXListCmdList = []
        cmdIdList = []
        newXList = ''
        nowCmd = ''
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
                i += 11
            i += 1
        mapXListCmdData[mapXListId] = mapXListCmdList
        newMapYList.append(newXList)
        mapXCmdIdData[mapXListId] = cmdIdList
    return {"Draw":newMapYList,"Cmd":mapXListCmdData,"CmdId":mapXCmdIdData}

def getPathList(rootPath:str) -> list:
    '''
    获取路径下所有子目录列表
    Keyword arguments:
    rootPath -- 要获取的目录所在根路径
    '''
    return [name for name in os.listdir(rootPath)
            if os.path.isdir(os.path.join(rootPath,name))
           ]

def initClothingData(originalClothingData):
    '''
    初始化服装类型数据
    '''
    clothingTypeData = {x:getOriginalClothing(j,k,originalClothingData[j][k][x]) for j in originalClothingData for k in originalClothingData[j] for x in originalClothingData[j][k]}
    CacheContorl.clothingTypeData = clothingTypeData

def getOriginalClothing(clothingType,clothingSex,nowClothingData):
    '''
    生成无分类的原始服装数据
    Keyword arguments:
    clothingType -- 服装类型
    clothingSex -- 服装性别倾向
    clothingData -- 原始服装数据
    '''
    nowClothingData['Type'] = clothingType
    nowClothingData['Sex'] = clothingSex
    return nowClothingData

def init():
    '''
    初始化游戏数据
    '''
    datapath = os.path.join(gamepath,'data')
    _loaddir(datapath)
