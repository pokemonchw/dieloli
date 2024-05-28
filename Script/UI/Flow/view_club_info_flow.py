
from Script.Design import handle_panel, constant
from Script.UI.Panel import view_club_info_panel
from Script.Config import normal_config

width = normal_config.config_normal.text_width
""" 屏幕宽度 """


@handle_panel.add_panel(constant.Panel.VIEW_CLUB_INFO)
def view_club_info():
    """查看角色状态监控面板"""
    now_panel = view_club_info_panel.ClubInfoPanel(width)
    now_panel.draw()
