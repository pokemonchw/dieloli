import os
from script.Core import GameData,GamePathConfig,JsonHandle

gamepath = GamePathConfig.gamepath

def loadConfigData() -> dict:
    '''
    读取游戏配置数据
    '''
    configPath = os.path.join(gamepath,'data','core_cfg.json')
    configData = JsonHandle._loadjson(configPath)
    return configData

def getFontData(fontId:str) -> dict:
    '''
    读取游戏字体样式配置数据
    Keyword arguments:
    listId -- 字体样式
    '''
    FontPath = os.path.join(gamepath,'data','FontConfig.json')
    FontData = JsonHandle._loadjson(FontPath)
    return FontData[fontId]

def getFontDataList() -> list:
    '''
    读取游戏字体样式配置列表
    '''
    FontPath = os.path.join(gamepath, 'data', 'FontConfig.json')
    FontData = JsonHandle._loadjson(FontPath)
    fontList = list(FontData.keys())
    return fontList

#配置数据定义
configData = loadConfigData()
game_name = configData['game_name']
verson = configData['verson']
author = configData['author']
verson_time = configData['verson_time']
background_color = configData['background_color']
language = configData['language']
window_width = configData['window_width']
window_hight = configData['window_hight']
textbox_width = configData['textbox_width']
textbox_hight = configData['textbox_hight']
text_width = int(configData['text_width'])
text_hight = int(configData['text_hight'])
inputbox_width = int(configData['inputbox_width'])
cursor = configData['cursor']
year = configData["year"]
month = configData["month"]
day = configData["day"]
hour = configData["hour"]
minute = configData["minute"]
max_save = configData['max_save']
save_page = configData['save_page']
characterlist_show = configData['characterlist_show']
text_wait = configData['text_wait']
home_url = configData['home_url']
licenses_url = configData['licenses_url']
random_npc_max = configData['random_npc_max']
proportion_teacher = configData['proportion_teacher']
proportion_student = configData['proportion_student']
threading_pool_max = configData['threading_pool_max']
in_scene_see_player_max = configData['insceneseeplayer_max']
see_character_clothes_max = configData['seecharacterclothes_max']
see_character_wearitem_max = configData['seecharacterwearitem_max']
