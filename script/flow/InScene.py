import core.CacheContorl as cache
import script.Panel.InScenePanel as inscenepanel
import core.game as game

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
    inputS = inputS + inscenepanel.seeMovePathPanel()
    yrn = game.askfor_All(inputS)

    pass