from Core import CacheContorl
from Design import MapHandle,GameTime
from Flow import InScene

# 主角移动
def playerMove(targetScene):
    playerPosition = CacheContorl.playObject['object']['0']['Position']
    intPlayerPosition = int(playerPosition)
    intTargetScene = int(targetScene)
    if intPlayerPosition == intTargetScene:
        InScene.getInScene_func()
    else:
        mapId = MapHandle.getMapIdForScene(playerPosition)
        nowSceneId = MapHandle.getMapSceneIdForSceneId(mapId,playerPosition)
        pathData = MapHandle.getPathfinding(mapId,nowSceneId,targetScene)
        if pathData == 'End':
            InScene.getInScene_func()
        else:
            targetScenePosition = MapHandle.getSceneIdForMapSceneId(mapId, targetScene)
            timeList = pathData['Time']
            MapHandle.playerMoveScene(playerPosition, targetScenePosition, '0')
            GameTime.setSubMinute(timeList[1])
            playerMove(targetScene)