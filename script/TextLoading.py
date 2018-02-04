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
attrTemplatePath = 'AttrTemplate.json'
systemTextPath = 'SystemText.json'
fontConfigPath = os.path.join(gamepath, 'data', 'FontConfig.json')

# 载入文本数据
def getData(jsonName):
    dataPath = os.path.join(gamepath,'data',language,jsonName)
    dataData = data._loadjson(dataPath)
    return dataData

# 获取messageAdv数据
def loadMessageAdv(advid):
    messageData = getData(messagePath)
    message = messageData[advid]
    return message

# 获取cmdadv数据
def loadCmdAdv(cmdid):
    cmdData = getData(cmdPath)
    cmdText = cmdData[cmdid]
    return cmdText

# 获取menu数据
def loadMenuText(menuid):
    menuData = getData(menuPath)
    menuText = menuData[menuid]
    return menuText

# 获取默认属性数据
def loadRoleAtrText(atrid):
    roleData = getData(rolePath)
    atrText = roleData[atrid]
    return atrText

# 获取舞台语文本
def loadStageWordText(wordId):
    stageWordData = getData(stageWordPath)
    stageWordText = stageWordData[wordId]
    return stageWordText

# 获取错误信息文本
def loadErrorText(errorId):
    errorData = getData(errorPath)
    errorText = errorData[errorId]
    return errorText

# 获取属性模板数据
def loadAttrTemplateText(temId):
    temData = getData(attrTemplatePath)
    temText = temData[temId]
    return temText

# 载入系统文本
def loadSystemText(systemId):
    systemData = getData(systemTextPath)
    systemText = systemData[systemId]
    return systemText

# 载入字体数据
def loadFontData(fontStyleName):
    fontList = getData(fontConfigPath)
    fontData = fontList[fontStyleName]
    return fontData