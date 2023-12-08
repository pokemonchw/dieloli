from Script.Design import handle_panel, constant
from Script.UI.Panel import game_setting_panel
from Script.Config import normal_config
from Script.Core import cache_control, game_type

width = normal_config.config_normal.text_width
""" 屏幕宽度 """
cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_panel.add_panel(constant.Panel.GAME_SETTING)
def game_setting_flow():
    """游戏设置面板"""
    now_panel = game_setting_panel.GameSettingPanel(width)
    while 1:
        if cache.now_panel_id != constant.Panel.GAME_SETTING:
            break
        now_panel.draw()
