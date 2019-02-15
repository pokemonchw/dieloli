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

def loadDirNow(dataPath):
    nowData = {}
    if os.listdir(dataPath):
        for i in os.listdir(dataPath):
            nowPath = os.path.join(dataPath,i)
            if os.path.isfile(nowPath):
                nowFile = i.split('.')
                if len(nowFile) > 1:
                    if nowFile[1] == 'json':
                        nowData[nowFile[0]] = _loadjson(nowPath)
                elif nowFile[0] == 'Map':
                    nowReadFile = open(nowPath,'r')
                    nowData['MapDraw'] = nowReadFile.read()
                    nowReadFile.close()
            else:
                nowData[i] = loadDirNow(nowPath)
    return nowData


# 获取路径下所有子路径列表
def getPathList(pathData):
    pathList = []
    for i in os.listdir(pathData):
        path = os.path.join(pathData,i)
        if os.path.isfile(path):
            pass
        else:
            pathList.append(i)
    return pathList

# 游戏初始化
def init():
    datapath = os.path.join(gamepath,'data')
    _loaddir(datapath)
