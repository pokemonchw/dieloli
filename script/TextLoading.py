import os

import core.GameConfig as config
import core.data as data
from core.pycfg import gamepath

cmdStartGameText = '1'
cmdLoadGameText = '2'
cmdQuitGameText = '3'

advGameLoadText = 'adv1'
advGameIntroduce = 'adv2'

menuRestart = '1'
menuQuit = '2'
menuOnTextBox = '3'
menuOffTextBox = '4'
menuSetting = '5'
menuAbout = '6'
menuFile = '7'
menuTest = '8'
menuOther = '9'

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