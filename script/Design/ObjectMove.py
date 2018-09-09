from script.Core import CacheContorl,TextLoading,EraPrint
from script.Design import MapHandle,GameTime,Update
from script.Flow import InScene

# 主角移动
def playerMove(targetScene):
    targetSceneId = MapHandle.getSceneIdForPath(targetScene)
    moveNow = objectMove('0',targetSceneId)
    if moveNow == 'Null':
        nullMessage = TextLoading.getTextData(TextLoading.messageId,'30')
        EraPrint.p(nullMessage)
        Update.gameUpdateFlow()
        InScene.getInScene_func()
    elif moveNow == 'End':
        Update.gameUpdateFlow()
        InScene.getInScene_func()
    else:
        playerMove(targetScene)

# 通用对象移动函数
def objectMove(objectId,targetSceneId):
    objectId = str(objectId)
    objectPosition = CacheContorl.playObject['object'][objectId]['Position']
    nowPositionPath = MapHandle.getScenePathForSceneId(objectPosition)
    targetPositionPath = MapHandle.getScenePathForSceneId(targetSceneId)
    sceneHierarchy = MapHandle.judgeSceneAffiliation(nowPositionPath,targetPositionPath)
    if sceneHierarchy == '0':
        mapPath = MapHandle.getCommonMapForScenePath(nowPositionPath,targetPositionPath)
        mapId = MapHandle.getMapIdForPath(mapPath)
        nowSceneId = MapHandle.getMapSceneIdForScenePath(mapId,nowPositionPath)
        targetSceneId = MapHandle.getMapSceneIdForScenePath(mapId,targetPositionPath)
        moveEnd = identicalMapMove(objectId,mapId,nowSceneId,targetSceneId)
    else:
        moveEnd = differenceMapMove(objectId,targetSceneId)
    return moveEnd

# 不同层级移动
def differenceMapMove(objectId,targetSceneId):
    objectPosition = CacheContorl.playObject['object'][objectId]['Position']
    nowPositionPath = MapHandle.getScenePathForSceneId(objectPosition)
    targetPositionPath = MapHandle.getScenePathForSceneId(targetSceneId)
    isAffiliation = MapHandle.judgeSceneIsAffiliation(nowPositionPath,targetPositionPath)
    nowTruePositionPath = MapHandle.getScenePathForTrue(nowPositionPath)
    if isAffiliation == '0':
        nowTrueAffiliation = MapHandle.judgeSceneIsAffiliation(nowTruePositionPath,targetPositionPath)
        if nowTrueAffiliation == '0':
            nowTrueMapId = MapHandle.getMapIdForScenePath(nowTruePositionPath)
            nowMapSceneId = MapHandle.getMapSceneIdForScenePath(nowTrueMapId,nowPositionPath)
            return identicalMapMove(objectId,nowTrueMapId,nowMapSceneId,'0')
        elif nowTrueAffiliation == '1':
            nowMapId = MapHandle.getMapIdForPath(targetPositionPath)
            nowMapSceneId = MapHandle.getMapSceneIdForScenePath(nowMapId,nowPositionPath)
            return identicalMapMove(objectId,nowMapId,nowMapSceneId,'0')
    elif isAffiliation == '1':
        nowMapId = MapHandle.getMapIdForPath(nowPositionPath)
        nowTargetMapSceneId = MapHandle.getMapSceneIdForScenePath(nowMapId,targetPositionPath)
        return identicalMapMove(objectId,nowMapId,'0',nowTargetMapSceneId)
    else:
        nowTrueMapPath = MapHandle.getMapForPath(nowTruePositionPath)
        nowTrueMapId = MapHandle.getMapIdForPath(nowTrueMapPath)
        if str(nowTrueMapId) == '0':
            nowTargetMapSceneId = MapHandle.getMapSceneIdForScenePath('0',targetPositionPath)
            nowMapSceneId = MapHandle.getMapSceneIdForScenePath('0',nowTruePositionPath)
            return identicalMapMove(objectId,'0',nowMapSceneId,nowTargetMapSceneId)
        else:
            relationMapList = MapHandle.getRelationMapListForScenePath(nowTruePositionPath)
            nowSceneRealMapPath = relationMapList[len(relationMapList) - 1]
            commonMap = MapHandle.getCommonMapForScenePath(nowTruePositionPath,targetPositionPath)
            nowSceneRealMapId = str(MapHandle.getMapIdForPath(nowSceneRealMapPath))
            commonMapId = str(MapHandle.getMapIdForPath(commonMap))
            realMapInMap = str(MapHandle.getMapIdForScenePath(nowSceneRealMapPath))
            targetMapSceneId = MapHandle.getMapSceneIdForScenePath(commonMapId,targetPositionPath)
            if nowSceneRealMapId == commonMapId:
                nowMapSceneId = MapHandle.getMapSceneIdForScenePath(commonMapId,nowTruePositionPath)
            elif realMapInMap == commonMapId:
                nowMapSceneId = MapHandle.getMapSceneIdForScenePath(commonMapId,nowSceneRealMapPath)
            identicalMapMove(objectId,commonMapId,nowMapSceneId,targetMapSceneId)

# 相同地图移动
def identicalMapMove(objectId,mapId,nowMapSceneId,targetMapSceneId):
    movePath = MapHandle.getPathfinding(mapId,nowMapSceneId,targetMapSceneId)
    if movePath != 'End' and movePath != 'Null':
        nowTargetSceneId = movePath['Path'][1]
        nowNeedTime = movePath['Time'][1]
        nowObjectPosition = MapHandle.getSceneIdForMapSceneId(mapId,nowMapSceneId)
        nowTargetPosition = MapHandle.getSceneIdForMapSceneId(mapId,nowTargetSceneId)
        MapHandle.playerMoveScene(nowObjectPosition,nowTargetPosition,objectId)
        GameTime.setSubMinute(nowNeedTime)
    return movePath
