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
def printMap(mapId,mapDir = ''):
    mapDir = os.path.join(mapDataDir, mapDir)
    mapPath = os.path.join(mapDir,mapId)
    openMap = open(mapPath)
    mapText = openMap.read()
    textStyle = richtext.setRichTextPrint(mapText,'standard')
    mapText = richtext.removeRichCache(mapText)
    sceneList = data.getPathList(mapDir)
    inputS = []
    mapStringList = mapText.split('\n')
    for mapString in mapStringList:
        for i in range(0, len(mapString)):
            if mapString[i] in sceneList:
                pycmd.pcmd(mapString[i], mapString[i], None)
                inputS.append(mapString[i])
            else:
                eprint.p(mapString[i], textStyle[i])
        eprint.p('\n')
    return inputS

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
    cache.sceneData = {"SceneData" : sceneData,"ScenePathData":scenePathData}