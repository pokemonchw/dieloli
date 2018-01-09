import os

import core.GameConfig as config
import core.data as data
from core.pycfg import gamepath

menuRestart = '1'
menuQuit = '2'
menuSetting = '3'
menuAbout = '4'
menuFile = '5'
menuOther = '6'

language = config.language

messagePath = 'MessageList.json'
cmdPath = 'CmdText.json'
menuPath = 'MenuText.json'
rolePath = 'RoleAttributes.json'
stageWordPath = 'StageWord.json'
errorPath = 'ErrorText.json'

def getData(jsonName):
    dataPath = os.path.join(gamepath,'data',language,jsonName)
    dataData = data._loadjson(dataPath)
    return dataData

def loadMessageAdv(advid):
    messageData = getData(messagePath)
    message = messageData[advid]
    return message

def loadCmdAdv(cmdid):
    cmdData = getData(cmdPath)
    cmdText = cmdData[cmdid]
    return cmdText

def loadMenuText(menuid):
    menuData = getData(menuPath)
    menuText = menuData[menuid]
    return menuText

def loadRoleAtrText(atrid):
    roleData = getData(rolePath)
    atrText = roleData[atrid]
    return atrText

def loadStageWordText(wordId):
    stageWordData = getData(stageWordPath)
    stageWordText = stageWordData[wordId]
    return stageWordText

def loadErrorText(errorId):
    errorData = getData(errorPath)
    errorText = errorData[errorId]
    return errorText