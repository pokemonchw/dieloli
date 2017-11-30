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

messagePath = os.path.join(gamepath, 'data',language,'MessageList.json')
messageData = data._loadjson(messagePath)
cmdPath = os.path.join(gamepath,'data',language,'CmdText.json')
cmdData = data._loadjson(cmdPath)
menuPath = os.path.join(gamepath,'data',language,'MenuText.json')
menuData = data._loadjson(menuPath)

def loadMessageAdv(advid):
    message = messageData[advid]
    return message

def loadCmdAdv(cmdid):
    cmdText = cmdData[cmdid]
    return cmdText

def loadMenuText(menuid):
    menuText = menuData[menuid]
    return menuText