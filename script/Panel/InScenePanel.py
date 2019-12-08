from script.Core import CacheContorl,TextLoading,EraPrint,PyCmd,GameConfig
from script.Design import GameTime,CmdButtonQueue,MapHandle,CharacterHandle,InputQueue,AttrText
import math

panelStateTextData = TextLoading.getTextData(TextLoading.cmdPath,'cmdSwitch')
panelStateOnText = panelStateTextData[1]
panelStateOffText = panelStateTextData[0]

def seeScenePanel():
    '''
    当前场景信息面板
    '''
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
    panelState = CacheContorl.panelState['SeeSceneCharacterListPage']
    switch = panelStateOnText
    if panelState == '0':
        switch = panelStateOffText
    sceneCharacterList = sceneData['SceneCharacterData']
    if len(sceneCharacterList) > 1:
        EraPrint.p(' ')
        PyCmd.pcmd(switch,'SeeSceneCharacterListPage')
    EraPrint.plittleline()

def seeSceneCharacterListPanel() -> list:
    '''
    当前场景角色列表面板
    '''
    inputS = []
    seeCharacterText = TextLoading.getTextData(TextLoading.messagePath,'26')
    EraPrint.p(seeCharacterText)
    EraPrint.p('\n')
    scenePath = CacheContorl.characterData['character']['0']['Position']
    scenePathStr = MapHandle.getMapSystemPathStrForList(scenePath)
    nameList = MapHandle.getSceneCharacterNameList(scenePathStr,True)
    nameList = getNowPageNameList(nameList)
    characterId = CacheContorl.characterData['characterId']
    characterData = CacheContorl.characterData['character'][characterId]
    characterName = characterData['Name']
    inputS = CmdButtonQueue.optionstr('',cmdColumn=10,cmdSize='center',askfor=False,cmdListData=nameList,nullCmd=characterName)
    return inputS

def changeSceneCharacterListPanel() -> list:
    '''
    当前场景角色列表页切换控制面板
    '''
    nameListMax = int(GameConfig.in_scene_see_player_max)
    nowPage = int(CacheContorl.panelState['SeeSceneCharacterListPanel'])
    scenePath = CacheContorl.characterData['character']['0']['Position']
    scenePathStr = MapHandle.getMapSystemPathStrForList(scenePath)
    sceneCharacterNameList = MapHandle.getSceneCharacterNameList(scenePathStr)
    characterMax = len(sceneCharacterNameList)
    pageMax = math.floor(characterMax / nameListMax)
    pageText = '(' + str(nowPage) + '/' + str(pageMax) + ')'
    inputS = CmdButtonQueue.optionint(CmdButtonQueue.changescenecharacterlist,cmdColumn=5,askfor=False,cmdSize='center')
    EraPrint.printPageLine(sample = '-',string = pageText)
    EraPrint.pl()
    return inputS

# 用于获取当前页面下的名字列表
def getNowPageNameList(nameList:list) -> list:
    '''
    获取当前角色列表页面角色姓名列表
    Keyword arguments:
    nameList -- 当前场景下角色列表
    '''
    nowPage = int(CacheContorl.panelState['SeeSceneCharacterListPanel'])
    nameListMax = int(GameConfig.in_scene_see_player_max)
    newNameList = []
    nowNameStartId = nowPage * nameListMax
    for i in range(nowNameStartId,nowNameStartId + nameListMax):
        if i < len(nameList):
            newNameList.append(nameList[i])
    return newNameList

def seeCharacterInfoPanel():
    '''
    查看当前互动对象信息面板
    '''
    characterInfo = TextLoading.getTextData(TextLoading.stageWordPath, '77')
    EraPrint.p(characterInfo)
    characterId = CacheContorl.characterData['characterId']
    characterData = CacheContorl.characterData['character'][characterId]
    characterName = characterData['Name']
    EraPrint.p(characterName)
    EraPrint.p(' ')
    sex = characterData['Sex']
    sexText = TextLoading.getTextData(TextLoading.stageWordPath, '2') + AttrText.getSexText(sex)
    EraPrint.p(sexText)
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
    stateText = AttrText.getStateText(characterId)
    EraPrint.p(' ')
    EraPrint.p(stateText)
    EraPrint.plittleline()

def jumpCharacterListPagePanel() -> str:
    '''
    角色列表页面跳转控制面板
    '''
    messageText = TextLoading.getTextData(TextLoading.messagePath,'32')
    nameListMax = int(GameConfig.in_scene_see_player_max)
    characterMax = CharacterHandle.getCharacterIndexMax()
    pageMax = math.floor(characterMax / nameListMax)
    EraPrint.p('\n' + messageText + '(0-' + str(pageMax) + ')')
    ans = InputQueue.waitInput(0,pageMax)
    EraPrint.p(ans)
    return ans

def inSceneButtonPanel(startId:int) -> list:
    '''
    场景页面基础控制菜单面板
    Keyword arguments:
    startId -- 基础控制菜单命令起始Id
    '''
    EraPrint.pline()
    inputs = CmdButtonQueue.optionint(cmdList=CmdButtonQueue.inscenelist1, cmdColumn=9, askfor=False, cmdSize='center',startId = startId)
    return inputs
