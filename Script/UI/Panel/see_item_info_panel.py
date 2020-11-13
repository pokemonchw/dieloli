from typing import Dict
from Script.Core import cache_contorl, text_handle
from Script.UI.Moudle import panel,draw
from Script.Config import game_config

class SeeCharacterItemBagPanel:
    """
    查看角色道具背包面板
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
        self.item_button_list:Dict[str,str] = {}
        """ 道具按钮返回列表 """
        self.old_page_return:str = ""
        """ 切换上一页的按钮返回 """
        self.next_page_return:str = ""
        """ 切换下一页的按钮返回 """
        character_data = cache_contorl.character_data[character_id]
        item_list = list(character_data.item)
        item_panel = panel.PageHandlePanel(item_list,ItemNameDraw,20,4,width,1,1,0,"","|")
        self.handle_panel = item_panel
        """ 页面控制对象 """

    def draw(self):
        self.handle_panel.draw()
        self.item_button_list = self.handle_panel.return_list
        self.old_page_return = self.handle_panel.old_page_return
        self.next_page_return = self.handle_panel.next_page_return

    def old_page(self):
        """ 切换上一页 """
        self.handle_panel.old_page()

    def next_page(self):
        """ 切换下一页 """
        self.handle_panel.next_page()


class ItemNameDraw:
    """
    按道具id绘制道具名
    Keyword arguments:
    text -- 道具的配表id
    width -- 最大宽度
    is_button -- 绘制按钮
    num_button -- 绘制数字按钮
    butoon_id -- 数字按钮的id
    """

    def __init__(self,text:str,width:int,is_button:bool,num_button:bool,button_id:int):
        """ 初始化绘制对象 """
        self.text:int = int(text)
        """ 道具的配表id """
        """ 道具名绘制文本 """
        self.width:int = width
        """ 最大宽度 """
        self.is_button:bool = is_button
        """ 绘制按钮 """
        self.num_button:bool = num_button
        """ 绘制数字按钮 """
        self.button_id:bool = button_id
        """ 数字按钮的id """
        self.button_return:str = str(button_id)
        """ 按钮返回值 """
        item_config = game_config.config_item[self.text]
        item_name = item_config.name
        self.draw_text:str = ""
        if is_button:
            if num_button:
                index_text = text_handle.id_index(button_id)
                self.draw_text = f"{index_text} {item_name}"
                self.button_return = str(button_id)
            else:
                self.draw_text = item_name
                self.button_return = item_name
        else:
            self.draw_text = f"[{item_name}]"

    def draw(self):
        """ 绘制道具 """
        if self.is_button:
            now_draw = draw.Button(self.draw_text,self.button_return)
        else:
            now_draw = draw.NormalDraw()
            now_draw.text = self.draw_text
        now_draw.max_width = self.width
        now_draw.draw()
