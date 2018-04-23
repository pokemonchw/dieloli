import core.CacheContorl as cache
import script.Panel.InScenePanel as inscenepanel

# 用于查看当前场景的流程
def seeScene_func():
    sceneData = cache.sceneData
    sceneId = cache.playObject['object']['0']['Position']
    inputS = []
    inscenepanel.seeScenePanel()
    scenePlayerList = sceneData['ScenePlayerData'][sceneId]
    if len(scenePlayerList) > 1:
        inputS.append(inscenepanel.seeScenePlayerListPanel())
    else:
        pass
    pass