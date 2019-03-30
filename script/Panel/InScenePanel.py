from script.Core import CacheContorl,TextLoading,EraPrint,PyCmd
from script.Design import GameTime,CmdButtonQueue,MapHandle

# 用于查看当前场景的面板
def seeScenePanel():
    titleText = TextLoading.getTextData(TextLoading.stageWordPath,'75')
    EraPrint.plt(titleText)
    timeText = GameTime.getDateText()
    EraPrint.p(timeText)
    EraPrint.p(' ')
    scenePath = CacheContorl.playObject['object']['0']['Position']
    scenePathStr = MapHandle.getMapSystemPathStrForList(scenePath)
    sceneData = CacheContorl.sceneData[scenePathStr].copy()
    sceneName = sceneData['SceneName']
    sceneInfoHead = TextLoading.getTextData(TextLoading.stageWordPath, '76')
    sceneInfo = sceneInfoHead + sceneName
    EraPrint.p(sceneInfo)
    EraPrint.plittleline()

# 用于查看当前场景上角色列表的面板
def seeScenePlayerListPanel():
    inputS = []
    seePlayerText = TextLoading.getTextData(TextLoading.messagePath,'26')
    EraPrint.p(seePlayerText)
    EraPrint.p('\n')
    scenePath = CacheContorl.playObject['object']['0']['Position']
    nameList = MapHandle.getScenePlayerNameList(scenePath,True)
    for name in nameList:
        PyCmd.pcmd(name, name, None)
        inputS.append(name)
        EraPrint.p(' ')
    EraPrint.plittleline()
    return inputS

# 用于查看对象信息的面板
def seeObjectInfoPanel():
    objectInfo = TextLoading.getTextData(TextLoading.stageWordPath, '77')
    EraPrint.p(objectInfo)
    objectId = CacheContorl.playObject['objectId']
    objectData = CacheContorl.playObject['object'][objectId]
    objectName = objectData['Name']
    EraPrint.p(objectName)
    EraPrint.p(' ')
    intimateInfo = TextLoading.getTextData(TextLoading.stageWordPath,'16')
    gracesInfo = TextLoading.getTextData(TextLoading.stageWordPath,'17')
    objectIntimate = objectData['Intimate']
    objectGraces = objectData['Graces']
    objectIntimateText = intimateInfo + objectIntimate
    objectGracesText = gracesInfo + objectGraces
    EraPrint.p(objectIntimateText)
    EraPrint.p(' ')
    EraPrint.p(objectGracesText)
    EraPrint.plittleline()

def inSceneButtonPanel():
    inputs = CmdButtonQueue.optionint(cmdList=CmdButtonQueue.inscenelist1, cmdColumn=9, askfor=False, cmdSize='center')
    EraPrint.plittleline()
    return inputs
