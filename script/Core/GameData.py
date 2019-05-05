# -*- coding: UTF-8 -*-
from script.Core.GamePathConfig import gamepath
import json

try:
    import cPickle as pickle
except ImportError:
    import pickle

import os

_gamedata = {}

# 初始化游戏数据
def gamedata():
    return _gamedata

# 判断文件编码是否为utf-8
def is_utf8bom(pathfile):
    if b'\xef\xbb\xbf' == open(pathfile, mode='rb').read(3):
        return True
    return False

# 载入json文件
def _loadjson(filepath):
    if is_utf8bom(filepath):
        ec='utf-8-sig'
    else:
        ec='utf-8'
    with open(filepath, 'r', encoding=ec) as f:
        try:
            jsondata = json.loads(f.read())
        except json.decoder.JSONDecodeError:
            print(filepath + '  无法读取，文件可能不符合json格式')
            jsondata = []
    return jsondata

# 载入路径下所有json文件
def _loaddir(dataPath):
    _gamedata.update(loadDirNow(dataPath))

sceneData = {}
mapData = {}
def loadDirNow(dataPath):
    nowData = {}
    if os.listdir(dataPath):
    # 上面这行可以删掉？
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
                            nowSceneData.update(_loadjson(nowPath))
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
                            nowMapData.update(_loadjson(nowPath))
                            mapData[mapSystemPathStr] = nowMapData
                        else:
                            nowData[nowFile[0]] = _loadjson(nowPath)
                elif nowFile[0] == 'Map':
                    pass # 同Map.json一起处理
                    # nowReadFile = open(nowPath,'r')
                    # nowMapData = {}
                    # nowMapData['MapDraw'] = nowReadFile.read()
                    # nowReadFile.close()
                    # nowMapSystemPath = getMapSystemPathForPath(nowPath)
                    # nowMapData['MapPath'] = nowMapSystemPath
                    # nowMapSystemPathStr = getMapSystemPathStr(nowMapSystemPath)
                    # mapData[nowMapSystemPathStr].update(nowMapData)
            else:
                nowData[i] = loadDirNow(nowPath)
    return nowData

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
