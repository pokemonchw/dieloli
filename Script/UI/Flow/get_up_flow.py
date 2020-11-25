from types import FunctionType
from Script.Core import get_text,constant
from Script.Design import handle_panel

_: FunctionType = get_text._
""" 翻译api """

@handle_panel.add_panel(constant.Panel.GET_UP)
def get_up_panel():
    """ 起床面板 """
