from types import FunctionType
from Script.Core import get_text,constant
from Script.Design import handle_panel

_:FunctionType = get_text._
""" 翻译api """

@handle_panel.add_panel(constant.Panel.CREATOR_CHARACTER)
def creator_character_panel():
    """ 创建角色面板 """
