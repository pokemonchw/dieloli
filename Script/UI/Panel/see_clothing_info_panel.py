from uuid import UUID
from types import FunctionType
from typing import List
from Script.Core import cache_contorl, game_type, text_handle, get_text
from Script.Config import game_config
from Script.UI.Moudle import draw

_: FunctionType = get_text._
""" 翻译api """

line_feed = draw.NormalDraw()
""" 换行绘制对象 """
line_feed.text = "\n"
line_feed.max_width = 1


class SeeCharacterPutOnClothingListPanel:
    """
    显示角色已穿戴服装面板
    Keyword arguments:
    character_id -- 角色id
    width -- 绘制宽度
    """

    def __init__(self,character_id:int,width:int):
        """ 初始化绘制对象 """
        self.character_id:int = character_id
        """ 绘制的角色id """
        self.width:int = width
        """ 最大绘制宽度 """

    def draw(self):
        """ 绘制面板 """
        character_data = cache_contorl.character_data[self.character_id]
        title_draw = draw.TitleLineDraw(_("人物服装"), self.width)
        title_draw.draw()
        draw_list = []
        id_width = 0
        for clothing_type in game_config.config_clothing_type:
            type_data = game_config.config_clothing_type[clothing_type]
            type_draw = draw.LittleTitleLineDraw(type_data.name,self.width,":")
            draw_list.append(type_draw)
            if clothing_type in character_data.put_on and isinstance(character_data.put_on[clothing_type],UUID):
                now_draw = ClothingInfoDrawPanel(self.character_id,clothing_type,character_data.put_on[clothing_type],self.width)
                now_id_width = text_handle.get_text_index(now_draw.text_list[0])
                if now_id_width > id_width:
                    id_width = now_id_width
            else:
                now_draw = draw.NormalDraw()
                now_draw.text = _("未穿戴")
                now_draw.max_width = self.width
            draw_list.append(now_draw)
            draw_list.append(line_feed)
        for value in draw_list:
            if "id_width" in value.__dict__:
                value.id_width = id_width
            value.draw()


class ClothingInfoDrawPanel:
    """
    服装信息绘制面板
    Keyword arguments:
    character_id -- 角色id
    clothing_type -- 服装类型
    clothing_type -- 服装id
    width -- 绘制宽度
    draw_button -- 是否按按钮绘制
    button_id -- 绘制按钮时的id
    """

    def __init__(self,character_id:int,clothing_type:int,clothing_id:UUID,width:int,draw_button:bool=False,button_id:int=0):
        """ 初始化绘制对象 """
        character_data = cache_contorl.character_data[character_id]
        self.clothing_data:game_type.Clothing = character_data.clothing[clothing_type][clothing_id]
        """ 当前服装数据 """
        self.width:int = width
        """ 最大绘制宽度 """
        self.draw_button:bool = draw_button
        """ 是否按按钮绘制 """
        self.button_id:int = button_id
        """ 绘制按钮时的id """
        self.id_width:int = 0
        """ 绘制时计算用的id宽度 """
        now_id_text = ""
        if self.draw_button:
            now_id_text = text_handle.id_index(self.button_id)
        fix_width = self.width - len(now_id_text)
        value_dict = {
            _("可爱"):self.clothing_data.sweet,
            _("性感"):self.clothing_data.sexy,
            _("帅气"):self.clothing_data.handsome,
            _("清新"):self.clothing_data.fresh,
            _("典雅"):self.clothing_data.elegant,
            _("清洁"):self.clothing_data.cleanliness,
            _("保暖"):self.clothing_data.warm,
        }
        describe_list = [
            _("可爱的"),
            _("性感的"),
            _("帅气的"),
            _("清新的"),
            _("典雅的"),
            _("清洁的"),
            _("保暖的")
        ]
        value_list = list(value_dict.values())
        describe_id = value_list.index(max(value_list))
        describe = describe_list[describe_id]
        clothing_config = game_config.config_clothing_tem[self.clothing_data.tem_id]
        clothing_name = f"{self.clothing_data.evaluation}{describe}{clothing_config.name}"
        fix_width -= text_handle.get_text_index(clothing_name)
        value_text = ""
        for value_id in value_dict:
            value = str(value_dict[value_id])
            if len(value) < 4:
                value = (4 - len(value)) * " " + value
            value_text += f"|{value_id}:{value}"
        value_text += "|"
        id_text = ""
        if self.draw_button:
            id_text = f"{now_id_text} {clothing_name}"
        else:
            id_text = clothing_name
        self.text_list:List[str] = [id_text,value_text]
        """ 绘制的文本列表 """

    def draw(self):
        self.text_list[1] = text_handle.align(self.text_list[1],"center",0,1,self.width-self.id_width)
        text_width = text_handle.get_text_index(self.text_list[0])
        if text_width < self.id_width:
            self.text_list[0] += " " * (self.id_width - text_width)
        now_text = f"{self.text_list[0]}{self.text_list[1]}"
        if self.draw_button:
            now_draw = draw.Button(now_text,str(self.button_id))
        else:
            now_draw = draw.NormalDraw()
            now_draw.text = now_text
        now_draw.max_width = self.width
        now_draw.draw()
