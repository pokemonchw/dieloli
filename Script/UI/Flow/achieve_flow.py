from Script.Design import handle_panel, constant
from Script.Config import normal_config
from Script.UI.Panel import achieve_panel

width = normal_config.config_normal.text_width
""" 屏幕宽度 """


@handle_panel.add_panel(constant.Panel.ACHIEVE)
def achieve_flow():
    """绘制成就面板"""
    now_panel = achieve_panel.AchievePanel(width)
    now_panel.draw()
