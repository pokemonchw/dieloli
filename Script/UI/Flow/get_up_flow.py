from types import FunctionType
from Script.Core import constant
from Script.Design import handle_panel
from Script.UI.Panel import get_up_panel
from Script.Config import normal_config

width = normal_config.config_normal.text_width
""" 屏幕宽度 """


@handle_panel.add_panel(constant.Panel.GET_UP)
def get_up_flow():
    """ 起床面板 """
    now_panel = get_up_panel.GetUpPanel(width)
    now_panel.draw()
