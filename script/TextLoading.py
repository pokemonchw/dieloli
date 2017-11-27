from core.pycfg import gamepath
import script.GameConfig as config
import core.data as data
import os

cmdStartGameText = '1'
cmdLoadGameText = '2'
cmdQuitGameText = '3'

advGameLoadText = 'adv1'
advGameIntroduce = 'adv2'

language = config.language

messagePath = os.path.join(gamepath, 'data',language,'MessageList.json')
messageData = data._loadjson(messagePath)
cmdPath = os.path.join(gamepath,'data',language,'CmdText.json')
cmdData = data._loadjson(cmdPath)

def loadMessageAdv(advid):
    message = messageData[advid]
    return message

def loadCmdAdv(cmdid):
    cmdText = cmdData[cmdid]
    return cmdText