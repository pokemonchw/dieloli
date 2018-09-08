from script.Core import CacheContorl,TextLoading,EraPrint
from script.Design import MapHandle,GameTime,Update
from script.Flow import InScene
import os

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
        print(1111111111)
        nowTrueAffiliation = MapHandle.judgeSceneIsAffiliation(nowTruePositionPath,targetPositionPath)
        if nowTrueAffiliation == '0':
            print(222222222222)
            nowTrueMapId = MapHandle.getMapIdForScenePath(nowTruePositionPath)
            nowMapSceneId = MapHandle.getMapSceneIdForScenePath(nowTrueMapId,nowPositionPath)
            return identicalMapMove(objectId,nowTrueMapId,nowMapSceneId,'0')
        elif nowTrueAffiliation == '1':
            print(3333333333333)
            nowMapId = MapHandle.getMapIdForPath(targetPositionPath)
            nowMapSceneId = MapHandle.getMapSceneIdForScenePath(nowMapId,nowPositionPath)
            return identicalMapMove(objectId,nowMapId,nowMapSceneId,'0')
    elif isAffiliation == '1':
        print(44444444444444)
        nowMapId = MapHandle.getMapIdForPath(nowPositionPath)
        nowTargetMapSceneId = MapHandle.getMapSceneIdForScenePath(nowMapId,targetPositionPath)
        return identicalMapMove(objectId,nowMapId,'0',nowTargetMapSceneId)
    else:
        print(5555555555555555)
        nowTrueMapId = MapHandle.getMapIdForScenePath(nowTruePositionPath)
        if str(nowTrueMapId) == '0':
            print(66666666666666)
            nowTargetMapSceneId = MapHandle.getMapSceneIdForScenePath('0',targetPositionPath)
            nowMapSceneId = MapHandle.getMapSceneIdForScenePath('0',nowTruePositionPath)
            return identicalMapMove(objectId,'0',nowMapSceneId,nowTargetMapSceneId)
        else:
            print(nowTrueMapId)
            print(777777777777777)
            nowMapSceneId = MapHandle.getMapSceneIdForScenePath(nowTrueMapId,nowTruePositionPath)
            return identicalMapMove(objectId,nowTrueMapId,nowMapSceneId,'0')

# 相同地图移动
def identicalMapMove(objectId,mapId,nowMapSceneId,targetMapSceneId):
    movePath = MapHandle.getPathfinding(mapId,nowMapSceneId,targetMapSceneId)
    if movePath == 'Null':
        return 'Null'
    elif movePath == 'End':
        return 'End'
    else:
        nowTargetSceneId = movePath['Path'][1]
        nowNeedTime = movePath['Time'][1]
        nowObjectPosition = MapHandle.getSceneIdForMapSceneId(mapId,nowMapSceneId)
        nowTargetPosition = MapHandle.getSceneIdForMapSceneId(mapId,nowTargetSceneId)
        MapHandle.playerMoveScene(nowObjectPosition,nowTargetPosition,objectId)
        GameTime.setSubMinute(nowNeedTime)
        return 'MoveEnd'

