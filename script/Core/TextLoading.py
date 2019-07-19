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

messagePath = 'MessageList'
cmdPath = 'CmdText'
menuPath = 'MenuText'
rolePath = 'RoleAttributes'
stageWordPath = 'StageWord'
errorPath = 'ErrorText'
attrTemplatePath = 'AttrTemplate'
systemTextPath = 'SystemText'
nameListPath = 'NameIndex'
familyNameListPath = 'FamilyIndex'
fontConfigPath = 'FontConfig'
barConfigPath = 'BarConfig'
phaseCourse = 'PhaseCourse'
course = 'Course'
courseSession = 'CourseSession'
knowledge = 'Knowledge'
languageSkillsPath = 'LanguageSkills'

# 获取文本数据
def getTextData(textPathId,textId):
    if textPathId in ['FontConfig','BarConfig']:
        return GameData._gamedata[textPathId][textId]
    else:
        return GameData._gamedata[language][textPathId][textId]

# 获取游戏数据
def getGameData(textPathId):
    return GameData._gamedata[language][textPathId]
