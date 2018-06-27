import os
from script.Core import GameData,ValueHandle,GamePathConfig

gamepath = GamePathConfig.gamepath

# 读取配置数据
def configData():
    configPath = os.path.join(gamepath,'data','core_cfg.json')
    configData = GameData._loadjson(configPath)
    return configData

# 读取字体样式配置数据
def getFontData(listId):
    FontPath = os.path.join(gamepath,'data','FontConfig.json')
    FontData = GameData._loadjson(FontPath)
    return FontData[listId]

# 读取字体样式配置列表
def getFontDataList():
    FontPath = os.path.join(gamepath, 'data', 'FontConfig.json')
    FontData = GameData._loadjson(FontPath)
    fontList = ValueHandle.dictKeysToList(FontData)
    return fontList

#配置数据定义
game_name = configData()['game_name']
verson = configData()['verson']
author = configData()['author']
verson_time = configData()['verson_time']
background_color = configData()['background_color']
language = configData()['language']
window_width = configData()['window_width']
window_hight = configData()['window_hight']
textbox_width = configData()['textbox_width']
textbox_hight = configData()['textbox_hight']
text_width = int(configData()['text_width'])
text_hight = int(configData()['text_hight'])
cursor = configData()['cursor']
year = configData()["year"]
month = configData()["month"]
day = configData()["day"]
hour = configData()["hour"]
minute = configData()["minute"]
max_save = configData()['max_save']
save_page = configData()['save_page']
playerlist_show = configData()['playerlist_show']
text_wait = configData()['text_wait']
home_url = configData()['home_url']
licenses_url = configData()['licenses_url']