from Script.Design import handle_panel, constant
from Script.UI.Panel import game_setting_panel
from Script.Config import normal_config

width = normal_config.config_normal.text_width
""" 屏幕宽度 """


@handle_panel.add_panel(constant.Panel.GAME_SETTING)
def game_setting_flow():
    """游戏设置面板"""
    now_panel = game_setting_panel.GameSettingPanel(width)
    now_panel.draw()
