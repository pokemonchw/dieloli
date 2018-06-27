import os
from script.Core import GameConfig,GameData,GamePathConfig

gamepath = GamePathConfig.gamepath

menuRestart = '1'
menuQuit = '2'
menuSetting = '3'
menuAbout = '4'
menuFile = '5'
menuOther = '6'

language = GameConfig.language

messagePath = 'MessageList.json'
cmdPath = 'CmdText.json'
menuPath = 'MenuText.json'
rolePath = 'RoleAttributes.json'
stageWordPath = 'StageWord.json'
errorPath = 'ErrorText.json'
attrTemplatePath = 'AttrTemplate.json'
systemTextPath = 'SystemText.json'
fontConfigPath = os.path.join(gamepath, 'data', 'FontConfig.json')
barConfigPath = os.path.join(gamepath,'data','BarConfig.json')

# 载入文本数据
def getData(jsonName):
    dataPath = os.path.join(gamepath,'data',language,jsonName)
    dataData = GameData._loadjson(dataPath)
    return dataData

messageId = 'message'
cmdId = 'cmd'
menuId = 'menu'
roleId = 'role'
stageWordId = 'stageWord'
errorId = 'error'
temId = 'tem'
systemId = 'system'
fontListId = 'fontList'
barListId = 'barList'

textDataList = {
    "message":getData(messagePath),"cmd":getData(cmdPath),"menu":getData(menuPath),
    "role":getData(rolePath),"stageWord":getData(stageWordPath),"error":getData(errorPath),
    "tem":getData(attrTemplatePath),"system":getData(systemTextPath),"fontList":getData(fontConfigPath),
    "barList":getData(barConfigPath)
}

# 获取文本数据
def getTextData(textPath,textId):
    textData = textDataList[textPath]
    return textData[textId]
