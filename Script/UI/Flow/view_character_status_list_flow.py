from Script.Design import handle_panel, constant
from Script.UI.Panel import view_character_status_list_panel
from Script.Config import normal_config

width = normal_config.config_normal.text_width
""" 屏幕宽度 """


@handle_panel.add_panel(constant.Panel.VIEW_CHARACTER_STATUS_LIST)
def view_character_status_list():
    """查看角色状态监控面板"""
    now_panel = view_character_status_list_panel.CharacterStatusListPanel(width)
    now_panel.draw()
