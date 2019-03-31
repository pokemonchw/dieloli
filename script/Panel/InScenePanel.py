from script.Core import CacheContorl,TextLoading,EraPrint,PyCmd
from script.Design import GameTime,CmdButtonQueue,MapHandle

# 用于查看当前场景的面板
def seeScenePanel():
    titleText = TextLoading.getTextData(TextLoading.stageWordPath,'75')
    EraPrint.plt(titleText)
    timeText = GameTime.getDateText()
    EraPrint.p(timeText)
    EraPrint.p(' ')
    scenePath = CacheContorl.characterData['character']['0']['Position']
    scenePathStr = MapHandle.getMapSystemPathStrForList(scenePath)
    sceneData = CacheContorl.sceneData[scenePathStr].copy()
    sceneName = sceneData['SceneName']
    sceneInfoHead = TextLoading.getTextData(TextLoading.stageWordPath, '76')
    sceneInfo = sceneInfoHead + sceneName
    EraPrint.p(sceneInfo)
    EraPrint.plittleline()

# 用于查看当前场景上角色列表的面板
def seeSceneCharacterListPanel():
    inputS = []
    seeCharacterText = TextLoading.getTextData(TextLoading.messagePath,'26')
    EraPrint.p(seeCharacterText)
    EraPrint.p('\n')
    scenePath = CacheContorl.characterData['character']['0']['Position']
    nameList = MapHandle.getSceneCharacterNameList(scenePath,True)
    for name in nameList:
        PyCmd.pcmd(name, name, None)
        inputS.append(name)
        EraPrint.p(' ')
    EraPrint.plittleline()
    return inputS

# 用于查看对象信息的面板
def seeCharacterInfoPanel():
    characterInfo = TextLoading.getTextData(TextLoading.stageWordPath, '77')
    EraPrint.p(characterInfo)
    characterId = CacheContorl.characterData['characterId']
    characterData = CacheContorl.characterData['character'][characterId]
    characterName = characterData['Name']
    EraPrint.p(characterName)
    EraPrint.p(' ')
    intimateInfo = TextLoading.getTextData(TextLoading.stageWordPath,'16')
    gracesInfo = TextLoading.getTextData(TextLoading.stageWordPath,'17')
    characterIntimate = characterData['Intimate']
    characterGraces = characterData['Graces']
    characterIntimateText = intimateInfo + characterIntimate
    characterGracesText = gracesInfo + characterGraces
    EraPrint.p(characterIntimateText)
    EraPrint.p(' ')
    EraPrint.p(characterGracesText)
    EraPrint.plittleline()

def inSceneButtonPanel():
    inputs = CmdButtonQueue.optionint(cmdList=CmdButtonQueue.inscenelist1, cmdColumn=9, askfor=False, cmdSize='center')
    EraPrint.plittleline()
    return inputs
