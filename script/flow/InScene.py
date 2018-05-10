import core.CacheContorl as cache
import script.Panel.InScenePanel as inscenepanel
import core.game as game
import design.MapHandle as maphandle
import flow.SeePlayerAttr as seeplayerattr
import flow.SeeMap as seemap
import core.PyCmd as pycmd

# 用于进入场景流程
def getInScene_func():
    sceneData = cache.sceneData
    sceneId = cache.playObject['object']['0']['Position']
    scenePlayerList = sceneData['ScenePlayerData'][sceneId]
    if len(scenePlayerList) > 1:
        cache.playObject['objectId'] = scenePlayerList[0]
        seeScene_func('0')
    else:
        seeScene_func('1')
    pass

# 用于查看当前场景的流程
def seeScene_func(judge):
    inputS = []
    inscenepanel.seeScenePanel()
    if judge  == '0':
        inputS = inputS + inscenepanel.seeScenePlayerListPanel()
    else:
        pass
    inscenepanel.seeObjectInfoPanel()
    inSceneCmdList1 = inscenepanel.inSceneButtonPanel()
    inputS = inputS + inSceneCmdList1
    startId1 = len(inSceneCmdList1)
    yrn = game.askfor_All(inputS)
    sceneId = cache.playObject['object']['0']['Position']
    scenePlayerNameList = maphandle.getScenePlayerNameList(sceneId)
    pycmd.clr_cmd()
    if yrn in scenePlayerNameList:
        seeplayerattr.seeAttrOnEveryTime_func('InScenePanel')
    elif yrn == '0':
        seemap.seeMapFlow()
    elif yrn == '1':
        seeplayerattr.seeAttrOnEveryTime_func('InScenePanel')
    elif yrn == '2':
        cache.playObject['objectId'] = '0'
        seeplayerattr.seeAttrOnEveryTime_func('InScenePanel')
    pass