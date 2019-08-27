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

def getTextData(textPathId,textId):
    '''
    按文件id和文本id读取指定文本数据
    Keyword arguments:
    textPathId -- 文件id
    textId -- 文件下的文本id
    '''
    if textPathId in ['FontConfig','BarConfig']:
        return GameData._gamedata[textPathId][textId]
    else:
        return GameData._gamedata[language][textPathId][textId]

def getGameData(textPathId):
    '''
    按文件id读取文件数据
    Keyword arguments:
    textPathId -- 文件id
    '''
    return GameData._gamedata[language][textPathId]

def getCharacterData(characterName):
    '''
    按角色名获取预设的角色模板数据
    Keyword arguments:
    characterName -- 角色名
    '''
    return GameData._gamedata[language]['character'][characterName]
