import design.MapHandle as maphandle
import core.CacheContorl as cache
import design.GameTime as gametime
import flow.InScene as inscene

# 主角移动
def playerMove(targetScene):
    playerPosition = cache.playObject['object']['0']['Position']
    intPlayerPosition = int(playerPosition)
    intTargetScene = int(targetScene)
    if intPlayerPosition == intTargetScene:
        inscene.getInScene_func()
    else:
        mapId = maphandle.getMapIdForScene(playerPosition)
        nowSceneId = maphandle.getMapSceneIdForSceneId(mapId,playerPosition)
        pathData = maphandle.getPathfinding(mapId,nowSceneId,targetScene)
        if pathData == 'End':
            inscene.getInScene_func()
        else:
            targetScenePosition = maphandle.getSceneIdForMapSceneId(mapId, targetScene)
            timeList = pathData['Time']
            maphandle.playerMoveScene(playerPosition, targetScenePosition, '0')
            gametime.setSubMinute(timeList[0])
            playerMove(targetScene)
    pass