from script.Core import CacheContorl,TextLoading,EraPrint
from script.Design import MapHandle,GameTime,Update

# 主角移动
def ownCharcterMove(targetScene:list):
    '''
    主角寻路至目标场景
    Keyword arguments:
    targetScene -- 寻路目标场景(在地图系统下的绝对坐标)
    '''
    moveNow = characterMove('0',targetScene)
    if moveNow == 'Null':
        nullMessage = TextLoading.getTextData(TextLoading.messagePath,'30')
        EraPrint.p(nullMessage)
    elif CacheContorl.characterData['character']['0'].Position != targetScene:
        ownCharcterMove(targetScene)
    Update.gameUpdateFlow()
    CacheContorl.characterData['characterId'] = '0'
    CacheContorl.nowFlowId = 'in_scene'

def characterMove(characterId:str,targetScene:list) -> 'MoveEnd:str_Null,str_End,list':
    '''
    通用角色移动控制
    Keyword arguments:
    characterId -- 角色id
    targetScene -- 寻路目标场景(在地图系统下的绝对坐标)
    '''
    nowPosition = CacheContorl.characterData['character'][characterId].Position
    sceneHierarchy = MapHandle.judgeSceneAffiliation(nowPosition,targetScene)
    if sceneHierarchy == 'common':
        mapPath = MapHandle.getCommonMapForScenePath(nowPosition,targetScene)
        nowMapSceneId = MapHandle.getMapSceneIdForScenePath(mapPath,nowPosition)
        targetMapSceneId = MapHandle.getMapSceneIdForScenePath(mapPath,targetScene)
        moveEnd = identicalMapMove(characterId,mapPath,nowMapSceneId,targetMapSceneId)
    else:
        moveEnd = differenceMapMove(characterId,targetScene)
    return moveEnd

def differenceMapMove(characterId:str,targetScene:list) -> 'MoveEnd:str_Null,str_End,list':
    '''
    角色跨地图层级移动
    Keyword arguments:
    characterId -- 角色id
    targetScene -- 寻路目标场景(在地图系统下的绝对坐标)
    '''
    nowPosition = CacheContorl.characterData['character'][characterId].Position
    isAffiliation = MapHandle.judgeSceneIsAffiliation(nowPosition,targetScene)
    nowTruePosition = MapHandle.getScenePathForTrue(nowPosition)
    mapDoorData = MapHandle.getMapDoorDataForScenePath(MapHandle.getMapSystemPathStrForList(nowTruePosition))
    doorScene = '0'
    nowTrueMap = MapHandle.getMapForPath(nowTruePosition)
    nowTrueMapMapSystemStr = MapHandle.getMapSystemPathStrForList(nowTrueMap)
    if isAffiliation == 'subordinate':
        nowTrueAffiliation = MapHandle.judgeSceneIsAffiliation(nowTruePosition,targetScene)
        if nowTrueAffiliation == 'subordinate':
            if mapDoorData != {}:
                doorScene = mapDoorData[nowTrueMapMapSystemStr]['Door']
            nowMapSceneId = MapHandle.getMapSceneIdForScenePath(nowTrueMap,nowPosition)
            return identicalMapMove(characterId,nowTrueMap,nowMapSceneId,doorScene)
        elif nowTrueAffiliation == 'superior':
            nowMap = MapHandle.getMapForPath(targetScene)
            nowMapSceneId = MapHandle.getMapSceneIdForScenePath(nowMap,nowPosition)
            return identicalMapMove(characterId,nowMap,nowMapSceneId,doorScene)
    else:
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
            else:
                nowMapSceneId =MapHandle.getMapSceneIdForScenePath(nowTrueMap,nowTruePosition)
                targetMapSceneId = '0'
            return identicalMapMove(characterId,commonMap,nowMapSceneId,targetMapSceneId)

def identicalMapMove(characterId:str,nowMap:list,nowMapSceneId:str,targetMapSceneId:str) -> 'MoveEnd:str_Null,str_End,list':
    '''
    角色在相同地图层级内移动
    Keyword arguments:
    characterId -- 角色id
    nowMap -- 当前地图路径
    nowMapSceneId -- 当前角色所在场景(当前地图层级下的相对坐标)
    targetMapSceneId -- 寻路目标场景(当前地图层级下的相对坐标)
    '''
    nowMapStr = MapHandle.getMapSystemPathStrForList(nowMap)
    movePath = MapHandle.getPathfinding(nowMapStr,nowMapSceneId,targetMapSceneId)
    if movePath != 'End' and movePath != 'Null':
        nowTargetSceneId = movePath['Path'][0]
        nowNeedTime = movePath['Time'][0]
        nowCharacterPosition = MapHandle.getScenePathForMapSceneId(nowMap,nowMapSceneId)
        nowTargetPosition = MapHandle.getScenePathForMapSceneId(nowMap,nowTargetSceneId)
        MapHandle.characterMoveScene(nowCharacterPosition,nowTargetPosition,characterId)
        GameTime.subTimeNow(nowNeedTime)
    return movePath
