from script.Core import RichText,EraPrint,PyCmd,CacheContorl,ValueHandle,TextHandle
import os,datetime

# 输出地图
def printMap(mapPath):
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

# 从地图路径获取地图图形
def getMapDrawForMapPath(mapPath):
    mapData = getMapDataForMapPath(mapPath)
    return mapData['MapDraw']

# 获取场景在地图上的位置
def getSceneIdInMapForScenePathOnMapPath(scenePath,mapPath):
    return scenePath[len(mapPath)]

# 查找场景所在地图
def getMapForPath(scenePath):
    mapPath = scenePath[:-1]
    mapPathStr = getMapSystemPathStrForList(mapPath)
    if mapPathStr in CacheContorl.mapData:
        return mapPath
    return getMapForPath(mapPath)

# 从地图路径获取地图数据
def getMapDataForMapPath(mapPath):
    mapData = CacheContorl.mapData.copy()
    if isinstance(mapPath,list):
        mapPath = getMapSystemPathStrForList(mapPath)
    mapData = mapData[mapPath]
    return mapData

# 获取地图下所有场景
def getSceneListForMap(mapPath):
    mapData = getMapDataForMapPath(mapPath)
    sceneList = list(mapData['PathEdge'].keys())
    return sceneList

# 获取地图下所有场景的名字
def getSceneNameListForMapPath(mapPath):
    sceneList = getSceneListForMap(mapPath)
    sceneNameData = {}
    for scene in sceneList:
        loadSceneData = getSceneDataForMap(mapPath,scene)
        sceneName = loadSceneData['SceneName']
        sceneNameData[scene] = sceneName
    return sceneNameData

# 场景移动
def characterMoveScene(oldScenePath,newScenePath,characterId):
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

# 从地图路径列表获取地图系统路径
def getMapSystemPathStrForList(nowList):
    if isinstance(nowList,list):
        return os.sep.join(nowList)
    return nowList

# 查询路径
def getPathfinding(mapPath,nowNode,targetNode):
    mapPathStr = getMapSystemPathStrForList(mapPath)
    if nowNode == targetNode:
        return 'End'
    else:
        return CacheContorl.mapData[mapPathStr]['SortedPath'][nowNode][str(targetNode)]

# 获取地图路径列表
def getSceneToSceneMapList(nowScenePath,targetScenePath):
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

# 查找节点共同所属地图
def getCommonMapForScenePath(sceneAPath,sceneBPath):
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

# 获取节点所属层级列表
def getMapHierarchyListForScenePath(nowScenePath,targetScenePath):
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

# 获取有效地图路径
def getMapPathForTrue(mapPath):
    mapPathStr = getMapSystemPathStrForList(mapPath)
    if mapPathStr in CacheContorl.mapData:
        return mapPath
    else:
        newMapPath = mapPath[:-1]
        return getMapPathForTrue(newMapPath)

# 判断场景所属关系
def judgeSceneIsAffiliation(nowScenePath,targetScenePath):
    if judgeSceneAffiliation(nowScenePath,targetScenePath) == 'subordinate':
        return 'subordinate'    #0
    elif judgeSceneAffiliation(targetScenePath,nowScenePath) == 'subordinate':
        return 'superior'   #1
    return 'common' #2

# 判断场景有无所属关系
def judgeSceneAffiliation(nowScenePath,targetScenePath):
    if nowScenePath[:-1] != targetScenePath[:-1]:
        if nowScenePath[:-1] != targetScenePath:
            if nowScenePath[:-1] != []:
                return judgeSceneAffiliation(nowScenePath[:-1],targetScenePath)
            else:
                return 'nobelonged'  #2
        else:
            return 'subordinate'  #1
    return 'common'  #0

# 获取场景所在所有直接地图位置
def getRelationMapListForScenePath(scenePath):
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

# 载入地图下对应场景数据
def getSceneDataForMap(mapPath,mapSceneId):
    mapPathStr = getMapSystemPathStrForList(mapPath)
    if mapPathStr == '':
        scenePathStr = str(mapSceneId)
    else:
        scenePathStr = mapPathStr + os.sep + str(mapSceneId)
    scenePath = getScenePathForTrue(scenePathStr)
    scenePathStr = getMapSystemPathStrForList(scenePath)
    return CacheContorl.sceneData[scenePathStr]

# 从对应地图场景id查找场景路径
def getScenePathForMapSceneId(mapPath,mapSceneId):
    newScenePath = mapPath.copy()
    newScenePath.append(mapSceneId)
    newScenePath = getScenePathForTrue(newScenePath)
    return newScenePath

# 从场景路径查找地图场景id
def getMapSceneIdForScenePath(mapPath,scenePath):
    return scenePath[len(mapPath)]

# 获取有效场景路径
def getScenePathForTrue(scenePath):
    scenePathStr = getMapSystemPathStrForList(scenePath)
    if scenePathStr in CacheContorl.sceneData:
        return scenePath
    else:
        if isinstance(scenePath,str):
            scenePath = scenePath.split(os.sep)
        scenePath.append('0')
        return getScenePathForTrue(scenePath)

# 从场景路径获取地图门数据
def getMapDoorDataForScenePath(scenePath):
    mapPath = getMapForPath(scenePath)
    return getMapDoorData(mapPath)

# 获取地图门数据
def getMapDoorData(mapPath):
    mapData = CacheContorl.mapData[mapPath]
    if "MapDoor" in  mapData:
        return mapData["MapDoor"]
    else:
        return {}

# 获取场景上所有角色的姓名列表
def getSceneCharacterNameList(scenePath,removeOwnCharacter = False):
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

# 获取场景上角色姓名对应角色id
def getCharacterIdByCharacterName(characterName,scenePath):
    characterNameList = getSceneCharacterNameList(scenePath)
    characterNameIndex = characterNameList.index(characterName)
    characterIdList = getSceneCharacterIdList(scenePath)
    return characterIdList[characterNameIndex]

# 获取场景上所有角色的id列表
def getSceneCharacterIdList(scenePath):
    scenePathStr = getMapSystemPathStrForList(scenePath)
    sceneCharacterData = CacheContorl.sceneData[scenePathStr]['SceneCharacterData']
    return sceneCharacterData

# 对场景上的角色按好感度进行排序
def sortSceneCharacterId(scenePath):
    scenePathStr = getMapSystemPathStrForList(scenePath)
    nowSceneCharacterIntimateData = {}
    for character in CacheContorl.sceneData[scenePathStr]['SceneCharacterData']:
        nowSceneCharacterIntimateData[character] = CacheContorl.characterData['character'][character]['Intimate']
    newSceneCharacterIntimateData = sorted(nowSceneCharacterIntimateData.items(),key=lambda x: (x[1],-int(x[0])),reverse=True)
    newSceneCharacterIntimateData = ValueHandle.twoBitArrayToDict(newSceneCharacterIntimateData)
    newCharacterList = list(newSceneCharacterIntimateData.keys())
    CacheContorl.sceneData[scenePathStr]['SceneCharacterData'] = newCharacterList
