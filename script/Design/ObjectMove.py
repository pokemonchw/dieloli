from script.Core import CacheContorl,TextLoading,EraPrint
from script.Design import MapHandle,GameTime,Update
from script.Flow import InScene
import datetime

# 主角移动
def playerMove(targetScene):
    moveNow = objectMove('0',targetScene)
    if moveNow == 'Null':
        nullMessage = TextLoading.getTextData(TextLoading.messagePath,'30')
        EraPrint.p(nullMessage)
        Update.gameUpdateFlow()
        InScene.getInScene_func()
    elif moveNow == 'End':
        Update.gameUpdateFlow()
        InScene.getInScene_func()
    else:
        playerMove(targetScene)

# 通用对象移动函数
def objectMove(objectId,targetScene):
    objectId = str(objectId)
    nowPosition = CacheContorl.playObject['object'][objectId]['Position']
    sceneHierarchy = MapHandle.judgeSceneAffiliation(nowPosition,targetScene)
    if sceneHierarchy == '0':
        mapPath = MapHandle.getCommonMapForScenePath(nowPosition,targetScene)
        nowMapSceneId = MapHandle.getMapSceneIdForScenePath(mapPath,nowPosition)
        targetMapSceneId = MapHandle.getMapSceneIdForScenePath(mapPath,targetScene)
        moveEnd = identicalMapMove(objectId,mapPath,nowMapSceneId,targetMapSceneId)
    else:
        moveEnd = differenceMapMove(objectId,targetScene)
    return moveEnd

# 不同层级移动
def differenceMapMove(objectId,targetScene):
    nowPosition = CacheContorl.playObject['object'][objectId]['Position']
    isAffiliation = MapHandle.judgeSceneIsAffiliation(nowPosition,targetScene)
    nowTruePosition = MapHandle.getScenePathForTrue(nowPosition)
    if isAffiliation == '0':
        nowTrueAffiliation = MapHandle.judgeSceneIsAffiliation(nowTruePosition,targetPositionPath)
        if nowTrueAffiliation == '0':
            nowTrueMap = MapHandle.getMapForPath(nowTruePosition)
            nowMapSceneId = MapHandle.getMapSceneIdForScenePath(nowTrueMap,nowPosition)
            return identicalMapMove(objectId,nowTrueMap,nowMapSceneId,'0')
        elif nowTrueAffiliation == '1':
            nowMap = MapHandle.getMapForPath(targetScene)
            nowMapSceneId = MapHandle.getMapSceneIdForScenePath(nowMap,nowPosition)
            return identicalMapMove(objectId,nowMap,nowMapSceneId,'0')
    elif isAffiliation == '1':
        nowMap = MapHandle.getMapForPath(nowPosition)
        nowTargetMapSceneId = MapHandle.getMapSceneIdForScenePath(nowMap,targetScene)
        return identicalMapMove(objectId,nowMapId,'0',nowTargetMapSceneId)
    else:
        nowTrueMap = MapHandle.getMapForPath(nowTruePosition)
        if nowTrueMap == []:
            nowTargetMapSceneId = MapHandle.getMapSceneIdForScenePath([],targetScene)
            nowMapSceneId = MapHandle.getMapSceneIdForScenePath([],nowTruePosition)
            return identicalMapMove(objectId,[],nowMapSceneId,nowTargetMapSceneId)
        else:
            relationMapList = MapHandle.getRelationMapListForScenePath(nowTruePosition)
            nowSceneRealMap = relationMapList[:-1]
            commonMap = MapHandle.getCommonMapForScenePath(nowTruePosition,targetScene)
            realMapInMap = MapHandle.getMapForPath(nowSceneRealMap)
            targetMapSceneId = MapHandle.getMapSceneIdForScenePath(commonMap,targetScene)
            if nowSceneRealMap == commonMap:
                nowMapSceneId = MapHandle.getMapSceneIdForScenePath(commonMap,nowTruePosition)
            elif realMapInMap == commonMap:
                nowMapSceneId = MapHandle.getMapSceneIdForScenePath(commonMap,nowSceneRealMap)
            identicalMapMove(objectId,commonMap,nowMapSceneId,targetMapSceneId)

# 相同地图移动
def identicalMapMove(objectId,nowMap,nowMapSceneId,targetMapSceneId):
    movePath = MapHandle.getPathfinding(nowMap,nowMapSceneId,targetMapSceneId)
    if movePath != 'End' and movePath != 'Null':
        nowTargetSceneId = movePath['Path'][1]
        nowNeedTime = movePath['Time'][1]
        nowObjectPosition = MapHandle.getScenePathForMapSceneId(nowMap,nowMapSceneId)
        nowTargetPosition = MapHandle.getScenePathForMapSceneId(nowMap,nowTargetSceneId)
        MapHandle.playerMoveScene(nowObjectPosition,nowTargetPosition,objectId)
        GameTime.subTimeNow(nowNeedTime)
    return movePath
