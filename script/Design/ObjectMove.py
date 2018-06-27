from script.Core import CacheContorl,TextLoading,EraPrint
from script.Design import MapHandle,GameTime,Update
from script.Flow import InScene

# 主角移动
def playerMove(targetScene):
    playerPosition = CacheContorl.playObject['object']['0']['Position']
    mapId = MapHandle.getMapIdForScene(playerPosition)
    nowSceneId = MapHandle.getMapSceneIdForSceneId(mapId, playerPosition)
    if nowSceneId == targetScene:
        Update.gameUpdateFlow()
        InScene.getInScene_func()
    else:
        pathData = MapHandle.getPathfinding(mapId,nowSceneId,targetScene)
        if pathData == 'End':
            Update.gameUpdateFlow()
            InScene.getInScene_func()
        elif pathData == 'Null':
            nullMessage = TextLoading.getTextData(TextLoading.messageId,'30')
            EraPrint.p(nullMessage)
            Update.gameUpdateFlow()
            InScene.getInScene_func()
        else:
            timeList = pathData['Time']
            pathList = pathData['Path']
            nowTargetScenePosition = MapHandle.getSceneIdForMapSceneId(mapId, pathList[1])
            MapHandle.playerMoveScene(playerPosition, nowTargetScenePosition, '0')
            GameTime.setSubMinute(timeList[1])
            playerMove(targetScene)