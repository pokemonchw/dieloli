from script.Core import CacheContorl,TextLoading,EraPrint
from script.Design import MapHandle,GameTime,Update

# 主角移动
def ownCharcterMove(targetScene):
    moveNow = characterMove('0',targetScene)
    if moveNow == 'Null':
        nullMessage = TextLoading.getTextData(TextLoading.messagePath,'30')
        EraPrint.p(nullMessage)
        Update.gameUpdateFlow()
        CacheContorl.nowFlowId = 'in_scene'
    elif moveNow == 'End':
        Update.gameUpdateFlow()
        CacheContorl.nowFlowId = 'in_scene'
    else:
        ownCharcterMove(targetScene)

# 通用对象移动函数
def characterMove(characterId,targetScene):
    characterId = str(characterId)
    nowPosition = CacheContorl.characterData['character'][characterId]['Position']
    sceneHierarchy = MapHandle.judgeSceneAffiliation(nowPosition,targetScene)
    if sceneHierarchy == '0':
        mapPath = MapHandle.getCommonMapForScenePath(nowPosition,targetScene)
        nowMapSceneId = MapHandle.getMapSceneIdForScenePath(mapPath,nowPosition)
        targetMapSceneId = MapHandle.getMapSceneIdForScenePath(mapPath,targetScene)
        moveEnd = identicalMapMove(characterId,mapPath,nowMapSceneId,targetMapSceneId)
    else:
        moveEnd = differenceMapMove(characterId,targetScene)
    return moveEnd

# 不同层级移动
def differenceMapMove(characterId,targetScene):
    nowPosition = CacheContorl.characterData['character'][characterId]['Position']
    isAffiliation = MapHandle.judgeSceneIsAffiliation(nowPosition,targetScene)
    nowTruePosition = MapHandle.getScenePathForTrue(nowPosition)
    if isAffiliation == '0':
        nowTrueAffiliation = MapHandle.judgeSceneIsAffiliation(nowTruePosition,targetScene)
        if nowTrueAffiliation == '0':
            nowTrueMap = MapHandle.getMapForPath(nowTruePosition)
            nowMapSceneId = MapHandle.getMapSceneIdForScenePath(nowTrueMap,nowPosition)
            return identicalMapMove(characterId,nowTrueMap,nowMapSceneId,'0')
        elif nowTrueAffiliation == '1':
            nowMap = MapHandle.getMapForPath(targetScene)
            nowMapSceneId = MapHandle.getMapSceneIdForScenePath(nowMap,nowPosition)
            return identicalMapMove(characterId,nowMap,nowMapSceneId,'0')
    elif isAffiliation == '1':
        nowMap = MapHandle.getMapForPath(nowPosition)
        nowTargetMapSceneId = MapHandle.getMapSceneIdForScenePath(nowMap,targetScene)
        return identicalMapMove(characterId,nowMap,'0',nowTargetMapSceneId)
    else:
        nowTrueMap = MapHandle.getMapForPath(nowTruePosition)
        if nowTrueMap == []:
            nowTargetMapSceneId = MapHandle.getMapSceneIdForScenePath([],targetScene)
            nowMapSceneId = MapHandle.getMapSceneIdForScenePath([],nowTruePosition)
            return identicalMapMove(characterId,[],nowMapSceneId,nowTargetMapSceneId)
        else:
            relationMapList = MapHandle.getRelationMapListForScenePath(nowTruePosition)
            nowSceneRealMap = relationMapList[-1]
            commonMap = MapHandle.getCommonMapForScenePath(nowTruePosition,targetScene)
            realMapInMap = MapHandle.getMapForPath(nowSceneRealMap)
            targetMapSceneId = MapHandle.getMapSceneIdForScenePath(commonMap,targetScene)
            if nowSceneRealMap == commonMap:
                nowMapSceneId = MapHandle.getMapSceneIdForScenePath(commonMap,nowTruePosition)
            elif realMapInMap == commonMap:
                nowMapSceneId = MapHandle.getMapSceneIdForScenePath(commonMap,nowSceneRealMap)
            identicalMapMove(characterId,commonMap,nowMapSceneId,targetMapSceneId)

# 相同地图移动
def identicalMapMove(characterId,nowMap,nowMapSceneId,targetMapSceneId):
    movePath = MapHandle.getPathfinding(nowMap,nowMapSceneId,targetMapSceneId)
    if movePath != 'End' and movePath != 'Null':
        nowTargetSceneId = movePath['Path'][1]
        nowNeedTime = movePath['Time'][1]
        nowCharacterPosition = MapHandle.getScenePathForMapSceneId(nowMap,nowMapSceneId)
        nowTargetPosition = MapHandle.getScenePathForMapSceneId(nowMap,nowTargetSceneId)
        MapHandle.characterMoveScene(nowCharacterPosition,nowTargetPosition,characterId)
        GameTime.subTimeNow(nowNeedTime)
    return movePath
