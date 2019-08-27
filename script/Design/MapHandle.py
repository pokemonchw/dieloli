from script.Core import EraPrint,PyCmd,CacheContorl,ValueHandle,TextHandle
import os

def printMap(mapPath):
    '''
    按地图路径绘制地图
    Ketword arguments:
    mapPath -- 地图路径
    '''
    mapDraw = getMapDrawForMapPath(mapPath)
    characterPosition = CacheContorl.characterData['character']['0']['Position']
    characterNowSceneId = getSceneIdInMapForScenePathOnMapPath(characterPosition,mapPath)
    inputS = []
    mapYList = mapDraw['Draw']
    mapXCmdListData = mapDraw['Cmd']
    mapXCmdIdListData = mapDraw['CmdId']
    for mapXListId in range(len(mapYList)):
        mapXList = mapYList[mapXListId]
        nowCmdList = mapXCmdListData[mapXListId]
        nowCmdIdList = mapXCmdIdListData[mapXListId]
        cmdListStr = ''.join(nowCmdList)
        EraPrint.p(TextHandle.align(mapXList + cmdListStr,'center',True),richTextJudge=False)
        i = 0
        while i in range(len(mapXList)):
            if nowCmdIdList != []:
                while i == nowCmdIdList[0]:
                    if nowCmdList[0] == characterNowSceneId:
                        EraPrint.p(nowCmdList[0],'nowmap',richTextJudge=False)
                        inputS.append(None)
                    else:
                        PyCmd.pcmd(nowCmdList[0], nowCmdList[0], None)
                        inputS.append(nowCmdList[0])
                    nowCmdList = nowCmdList[1:]
                    nowCmdIdList = nowCmdIdList[1:]
                    if nowCmdList == []:
                        break
                if nowCmdIdList != []:
                    EraPrint.p(mapXList[i:nowCmdIdList[0]])
                    i = nowCmdIdList[0]
                else:
                    EraPrint.p(mapXList[i:])
                    i = len(mapXList)
            else:
                EraPrint.p(mapXList[i:])
                i = len(mapXList)
        EraPrint.p('\n')
    return inputS

def getMapDrawForMapPath(mapPath):
    '''
    从地图路径获取地图绘制数据
    Keyword arguments:
    mapPath -- 地图路径
    '''
    mapData = getMapDataForMapPath(mapPath)
    return mapData['MapDraw']

def getSceneIdInMapForScenePathOnMapPath(scenePath,mapPath):
    '''
    获取场景在地图上的相对位置
    Keyword arguments:
    scenePath -- 场景路径
    mapPath -- 地图路径
    '''
    return scenePath[len(mapPath)]

def getMapForPath(scenePath):
    '''
    查找场景所在地图路径
    Keyword arguments:
    scenePath -- 场景路径
    '''
    mapPath = scenePath[:-1]
    mapPathStr = getMapSystemPathStrForList(mapPath)
    if mapPathStr in CacheContorl.mapData:
        return mapPath
    return getMapForPath(mapPath)

def getMapDataForMapPath(mapPath):
    '''
    从地图路径获取地图数据
    Keyword arguments:
    mapPath -- 地图路径
    '''
    mapData = CacheContorl.mapData.copy()
    if isinstance(mapPath,list):
        mapPath = getMapSystemPathStrForList(mapPath)
    mapData = mapData[mapPath]
    return mapData

def getSceneListForMap(mapPath):
    '''
    获取地图下所有场景
    Keyword arguments:
    mapPath -- 地图路径
    '''
    mapData = getMapDataForMapPath(mapPath)
    sceneList = list(mapData['PathEdge'].keys())
    return sceneList

def getSceneNameListForMapPath(mapPath):
    '''
    获取地图下所有场景的名字
    Keyword arguments:
    mapPath -- 地图路径
    '''
    sceneList = getSceneListForMap(mapPath)
    sceneNameData = {}
    for scene in sceneList:
        loadSceneData = getSceneDataForMap(mapPath,scene)
        sceneName = loadSceneData['SceneName']
        sceneNameData[scene] = sceneName
    return sceneNameData

def characterMoveScene(oldScenePath,newScenePath,characterId):
    '''
    将角色移动至新场景
    Keyword arguments:
    oldScenePath -- 旧场景路径
    newScenePath -- 新场景路径
    characterId -- 角色id
    '''
    oldScenePathStr = getMapSystemPathStrForList(oldScenePath)
    newScenePathStr = getMapSystemPathStrForList(newScenePath)
    oldSceneCharacterData = CacheContorl.sceneData[oldScenePathStr]["SceneCharacterData"]
    newSceneCharacterData = CacheContorl.sceneData[newScenePathStr]["SceneCharacterData"]
    if characterId in oldSceneCharacterData:
        del oldSceneCharacterData[characterId]
        CacheContorl.sceneData[oldScenePathStr]["SceneCharacterData"] = oldSceneCharacterData
    if characterId not in newSceneCharacterData:
        CacheContorl.characterData['character'][characterId]['Position'] = newScenePath
        newSceneCharacterData[characterId] = 0
        CacheContorl.sceneData[newScenePathStr]["SceneCharacterData"] = newSceneCharacterData

def getMapSystemPathStrForList(nowList):
    '''
    将地图路径列表数据转换为字符串
    Keyword arguments:
    nowList -- 地图路径列表数据
    '''
    if isinstance(nowList,list):
        return os.sep.join(nowList)
    return nowList

def getPathfinding(mapPath,nowNode,targetNode):
    '''
    查询寻路路径
    Keyword arguments:
    mapPath -- 地图路径
    nowNode -- 当前节点相对位置
    targetNode -- 目标节点相对位置
    '''
    mapPathStr = getMapSystemPathStrForList(mapPath)
    if nowNode == targetNode:
        return 'End'
    else:
        return CacheContorl.mapData[mapPathStr]['SortedPath'][nowNode][str(targetNode)]

def getSceneToSceneMapList(nowScenePath,targetScenePath):
    '''
    获取场景到场景之间需要经过的地图列表
    如果两个场景属于同一地图并在同一层级，则返回common
    Keyword arguments:
    nowScenePath -- 当前场景路径
    targetScenePath -- 目标场景路径
    '''
    sceneAffiliation = judgeSceneAffiliation(nowScenePath,targetScenePath)
    if sceneAffiliation == 'common':
        return 'common'
    elif sceneAffiliation == 'subordinate':
        return getMapHierarchyListForScenePath(nowScenePath,targetScenePath)
    elif sceneAffiliation == 'nobelonged':
        commonMap = getCommonMapForScenePath(nowScenePath,targetScenePath)
        nowSceneToCommonMap = getMapHierarchyListForScenePath(nowScenePath,commonMap)
        targetSceneToCommonMap = getMapHierarchyListForScenePath(targetScenePath,commonMap)
        commonMapToTargetScene = ValueHandle.reverseArrayList(targetSceneToCommonMap)
        return nowSceneToCommonMap + commonMapToTargetScene[1:]

def getCommonMapForScenePath(sceneAPath,sceneBPath):
    '''
    查找场景共同所属地图
    Keyword arguments:
    sceneAPath -- 场景A路径
    sceneBPath -- 场景B路径
    '''
    hierarchy = []
    if sceneAPath[:-1] == [] or sceneBPath[:-1] == []:
        return hierarchy
    else:
        for i in range(0,len(sceneAPath)):
            try:
                if sceneAPath[i] == sceneBPath[i]:
                    hierarchy.append(sceneAPath[i])
                else:
                    break
            except IndexError:
                break
        return getMapPathForTrue(hierarchy)

def getMapHierarchyListForScenePath(nowScenePath,targetScenePath):
    '''
    查找当前场景到目标场景之间的层级列表(仅当当前场景属于目标场景的子场景时可用)
    Keyword arguments:
    nowScenePath -- 当前场景路径
    targetScenePath -- 目标场景路径
    '''
    hierarchyList = []
    nowPath = None
    while(True):
        if nowPath == None:
            nowPath = nowScenePath[-1]
        if nowPath != targetScenePath:
            hierarchyList.append(nowPath)
            nowPath = nowPath[-1]
        else:
            break
    return hierarchyList

def getMapPathForTrue(mapPath):
    '''
    判断地图路径是否是有效的地图路径，若不是，则查找上层路径，直到找到有效地图路径并返回
    Keyword arguments:
    mapPath -- 当前地图路径
    '''
    mapPathStr = getMapSystemPathStrForList(mapPath)
    if mapPathStr in CacheContorl.mapData:
        return mapPath
    else:
        newMapPath = mapPath[:-1]
        return getMapPathForTrue(newMapPath)

def judgeSceneIsAffiliation(nowScenePath,targetScenePath):
    '''
    获取场景所属关系
    当前场景属于目标场景的子场景 -> 返回'subordinate'
    目标场景属于当前场景的子场景 -> 返回'superior'
    other -> 返回'common'
    Keyword arguments:
    nowScenePath -- 当前场景路径
    targetScenePath -- 目标场景路径
    '''
    if judgeSceneAffiliation(nowScenePath,targetScenePath) == 'subordinate':
        return 'subordinate'
    elif judgeSceneAffiliation(targetScenePath,nowScenePath) == 'subordinate':
        return 'superior'
    return 'common'

def judgeSceneAffiliation(nowScenePath,targetScenePath):
    '''
    判断场景有无所属关系
    当前场景属于目标场景的子场景 -> 返回'subordinate'
    当前场景与目标场景的第一个上级场景相同 -> 返回'common'
    other -> 返回'nobelonged'
    Keyword arguments:
    nowScenePath -- 当前场景路径
    targetScenePath -- 目标场景路径
    '''
    if nowScenePath[:-1] != targetScenePath[:-1]:
        if nowScenePath[:-1] != targetScenePath:
            if nowScenePath[:-1] != []:
                return judgeSceneAffiliation(nowScenePath[:-1],targetScenePath)
            else:
                return 'nobelonged'  #2
        else:
            return 'subordinate'  #1
    return 'common'  #0

def getRelationMapListForScenePath(scenePath):
    '''
    获取场景所在所有直接地图(当前场景id为0，所在地图在上层地图相对位置也为0，视为直接地图)位置
    Keyword arguments:
    scenePath -- 当前场景路径
    '''
    nowPath = scenePath
    nowMapPath = scenePath[:-1]
    nowPathId = nowPath[-1]
    mapList = []
    if nowMapPath != [] and nowMapPath[:-1] != []:
        mapList.append(nowMapPath)
        if nowPathId == '0':
            return mapList + getRelationMapListForScenePath(nowMapPath)
        else:
            return mapList
    else:
        mapList.append(nowMapPath)
        return mapList

def getSceneDataForMap(mapPath,mapSceneId):
    '''
    载入地图下对应场景数据
    Keyword arguments:
    mapPath -- 地图路径
    mapSceneId -- 场景相对位置
    '''
    mapPathStr = getMapSystemPathStrForList(mapPath)
    if mapPathStr == '':
        scenePathStr = str(mapSceneId)
    else:
        scenePathStr = mapPathStr + os.sep + str(mapSceneId)
    scenePath = getScenePathForTrue(scenePathStr)
    scenePathStr = getMapSystemPathStrForList(scenePath)
    return CacheContorl.sceneData[scenePathStr]

def getScenePathForMapSceneId(mapPath,mapSceneId):
    '''
    从场景在地图中的相对位置获取场景路径
    Keyword arguments:
    mapPath -- 地图路径
    mapSceneId -- 场景在地图中的相对位置
    '''
    newScenePath = mapPath.copy()
    newScenePath.append(mapSceneId)
    newScenePath = getScenePathForTrue(newScenePath)
    return newScenePath

def getMapSceneIdForScenePath(mapPath,scenePath):
    '''
    从场景路径查找场景在地图中的相对位置
    Keyword arguments:
    mapPath -- 地图路径
    scenePath -- 场景路径
    '''
    return scenePath[len(mapPath)]

def getScenePathForTrue(scenePath):
    '''
    获取场景的有效路径(当前路径下若不存在场景数据，则获取当前路径下相对位置为0的路径)
    Keyword arguments:
    scenePath -- 场景路径
    '''
    scenePathStr = getMapSystemPathStrForList(scenePath)
    if scenePathStr in CacheContorl.sceneData:
        return scenePath
    else:
        if isinstance(scenePath,str):
            scenePath = scenePath.split(os.sep)
        scenePath.append('0')
        return getScenePathForTrue(scenePath)

def getMapDoorDataForScenePath(scenePath):
    '''
    从场景路径获取当前地图到其他地图的门数据
    Keyword arguments:
    scenePath -- 场景路径
    '''
    mapPath = getMapForPath(scenePath)
    return getMapDoorData(mapPath)

def getMapDoorData(mapPath):
    '''
    获取地图下通往其他地图的门数据
    Keyword arguments:
    mapPath -- 地图路径
    '''
    mapData = CacheContorl.mapData[mapPath]
    if "MapDoor" in  mapData:
        return mapData["MapDoor"]
    else:
        return {}

def getSceneCharacterNameList(scenePath,removeOwnCharacter = False):
    '''
    获取场景上所有角色的姓名列表
    Keyword arguments:
    scenePath -- 场景路径
    removeOwnCharacter -- 从姓名列表中移除主角 (default False)
    '''
    scenePathStr = getMapSystemPathStrForList(scenePath)
    sceneCharacterData = CacheContorl.sceneData[scenePathStr]['SceneCharacterData']
    nowSceneCharacterList = sceneCharacterData.copy()
    nameList = []
    if removeOwnCharacter:
        nowSceneCharacterList.remove('0')
    for characterId in nowSceneCharacterList:
        characterName = CacheContorl.characterData['character'][str(characterId)]['Name']
        nameList.append(characterName)
    return nameList

def getCharacterIdByCharacterName(characterName,scenePath):
    '''
    获取场景上角色姓名对应的角色id
    Keyword arguments:
    characterName -- 角色姓名
    scenePath -- 场景路径
    '''
    characterNameList = getSceneCharacterNameList(scenePath)
    characterNameIndex = characterNameList.index(characterName)
    characterIdList = getSceneCharacterIdList(scenePath)
    return characterIdList[characterNameIndex]

def getSceneCharacterIdList(scenePath):
    '''
    获取场景上所有角色的id列表
    Keyword arguments:
    scenePath -- 场景路径
    '''
    scenePathStr = getMapSystemPathStrForList(scenePath)
    sceneCharacterData = CacheContorl.sceneData[scenePathStr]['SceneCharacterData']
    return sceneCharacterData

def sortSceneCharacterId(scenePath):
    '''
    对场景上的角色按好感度进行排序
    Keyword arguments:
    scenePath -- 场景路径
    '''
    scenePathStr = getMapSystemPathStrForList(scenePath)
    nowSceneCharacterIntimateData = {}
    for character in CacheContorl.sceneData[scenePathStr]['SceneCharacterData']:
        nowSceneCharacterIntimateData[character] = CacheContorl.characterData['character'][character]['Intimate']
    newSceneCharacterIntimateData = sorted(nowSceneCharacterIntimateData.items(),key=lambda x: (x[1],-int(x[0])),reverse=True)
    newSceneCharacterIntimateData = ValueHandle.twoBitArrayToDict(newSceneCharacterIntimateData)
    newCharacterList = list(newSceneCharacterIntimateData.keys())
    CacheContorl.sceneData[scenePathStr]['SceneCharacterData'] = newCharacterList
