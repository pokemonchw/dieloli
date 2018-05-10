import core.CacheContorl as cache
import core.TextLoading as textload
import core.EraPrint as eprint
import core.PyCmd as pycmd
import design.GameTime as gametime
import design.Ans as ans

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
        if playerId == '0':
            pass
        else:
            playerName = cache.playObject['object'][str(playerId)]['Name']
            pycmd.pcmd(playerName, playerName, None)
            inputS.append(playerName)
            eprint.p(' ')
    eprint.plittleline()
    return inputS

# 用于查看对象信息的面板
def seeObjectInfoPanel():
    objectInfo = textload.getTextData(textload.stageWordId, '77')
    eprint.p(objectInfo)
    objectId = cache.playObject['objectId']
    objectData = cache.playObject['object'][objectId]
    objectName = objectData['Name']
    eprint.p(objectName)
    eprint.p(' ')
    intimateInfo = textload.getTextData(textload.stageWordId,'16')
    gracesInfo = textload.getTextData(textload.stageWordId,'17')
    objectIntimate = objectData['Intimate']
    objectGraces = objectData['Graces']
    objectIntimateText = intimateInfo + objectIntimate
    objectGracesText = gracesInfo + objectGraces
    eprint.p(objectIntimateText)
    eprint.p(' ')
    eprint.p(objectGracesText)
    eprint.plittleline()

def inSceneButtonPanel():
    inputs = ans.optionint(cmdList=ans.inscenelist1,cmdColumn=9,askfor=False,cmdSize='center')
    eprint.plittleline()
    return inputs