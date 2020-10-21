from typing import Dict
from types import FunctionType
from Script.UI.Moudle import draw
from Script.Core import cache_contorl,get_text

panel_info_data = {
}

_:FunctionType = get_text._
""" 翻译api """

class SeeCharacterInfoPanel:
    """ 用于查看角色属性的面板对象 """

    def __init__(self,character_id:int):
        """
        初始化绘制对象
        Keyword arguments:
        character_id -- 角色id
        """
        self.button_list:List[draw.Button] = []
        """ 控制切换属性页的按钮列表 """
        self.return_list:Dict[str,str] = {}
        """
        切换属性页按钮的响应列表
        按钮返回值:按钮文本
        """
        self.max_width = 0
        """ 绘制的最大宽度 """
        self.now_panel = 0
        """ 当前的属性页id """
        self.character_id = character_id
        """ 要绘制的角色id """

    def draw(self):
        pass

class CharacterInfoHead:
    """ 角色信息面板头部面板 """

    def __init__(self,character_id:int,width int):
        """
        初始化绘制对象
        Keyword arguments:
        character_id -- 角色id
        width -- 最大宽度
        """
        self.character_id = character_id
        """ 要绘制的角色id """
        character_data = cache_contorl.character_data[character_id]
        message = _(f"No.{character_id} 姓名:{character_data.name} 称呼:{character_data.nick_name}")
        message_draw = draw.CenterDraw()
        message_draw.max_width = width / 2
        message_draw.text = message
        hp_draw = draw.InfoBarDraw()
        hp_draw.max_width = width / 2
        hp_draw.set("hp_bar",character_data.hit_point_max,character_data.hit_point,_("体力"))
        mp_draw.set("mp_bar",character_data.mana_point_max,character_data.mana_point,_("气力"))

    def draw(self):
        """ 绘制面板 """
