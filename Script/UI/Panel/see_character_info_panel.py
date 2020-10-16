from Script.Design import handle_panel
from Script.Core import constant

@handle_panel.add_panel(constant.Panel.SEE_CHARACTER_INFO)
def see_character_info_panel():
    """ 查看角色属性面板 """
