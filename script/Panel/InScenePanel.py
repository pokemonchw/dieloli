import core.CacheContorl as cache
import core.TextLoading as textload
import core.EraPrint as eprint
import script.GameTime as gametime
import core.PyCmd as pycmd

# 用于查看当前场景的面板
def seeScenePanel():
    sceneData = cache.sceneData
    titleText = textload.getTextData(textload.stageWordId,'75')
    eprint.plt(titleText)
    timeText = gametime.getDateText()
    eprint.p(timeText)
    eprint.p(' ')
    sceneIdS = cache.playObject['object']['0']['Position']
    sceneId = int(sceneIdS)
    sceneName = sceneData['SceneData'][sceneId]['SceneName']
    sceneInfoHead = textload.getTextData(textload.stageWordId, '76')
    sceneInfo = sceneInfoHead + sceneName
    eprint.p(sceneInfo)
    eprint.plittleline()

# 用于查看当前场景上角色列表的面板
def seeScenePlayerListPanel():
    inputS = []
    sceneData = cache.sceneData
    seePlayerText = textload.getTextData(textload.messageId,'26')
    eprint.p(seePlayerText)
    eprint.p('\n')
    sceneIdS = cache.playObject['object']['0']['Position']
    scenePlayerList = sceneData['ScenePlayerData'][sceneIdS]
    for playerId in scenePlayerList:
        playerName = cache.playObject['object'][str(playerId)]['Name']
        pycmd.pcmd(playerName,playerName,None)
        inputS.append(playerName)
        eprint.p(' ')
    eprint.plittleline()
    return inputS