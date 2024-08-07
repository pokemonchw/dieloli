from Script.Design import handle_panel, constant
from Script.UI.Panel import view_club_list_panel
from Script.Config import normal_config

width = normal_config.config_normal.text_width
""" 屏幕宽度 """


@handle_panel.add_panel(constant.Panel.VIEW_CLUB_LIST)
def view_club_list():
    """查看角色状态监控面板"""
    now_panel = view_club_list_panel.ClubListPanel(width)
    now_panel.draw()
