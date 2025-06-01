from types import FunctionType
from Script.Core import (
    cache_control,
    game_type,
    text_handle,
    get_text,
    flow_handle,
    py_cmd,
)
from Script.Design import map_handle, constant
from Script.Config import game_config, normal_config
from Script.UI.Model import panel, draw

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
_: FunctionType = get_text._
""" 翻译api """
window_width: int = normal_config.config_normal.text_width
""" 窗体宽度 """
line_feed = draw.NormalDraw()
""" 换行绘制对象 """
line_feed.text = "\n"
line_feed.width = 1


class ItemShopPanel:
    """
    用于查看道具商店界面面板对象
    Keyword arguments:
    width -- 绘制宽度
    """

    def __init__(self, width: int):
        """初始化绘制对象"""
        self.width: int = width
        """ 绘制的最大宽度 """

    def draw(self):
        """绘制对象"""
        character_data: game_type.Character = cache.character_data[0]
        scene_position = character_data.position
        scene_position_str = map_handle.get_map_system_path_str_for_list(scene_position)
        scene_name = cache.scene_data[scene_position_str].scene_name
        title_draw = draw.TitleLineDraw(scene_name, self.width)
        handle_panel = panel.PageHandlePanel([], BuyItemByItemNameDraw, 10, 5, self.width, 1, 1, 0)
        while 1:
            return_list = []
            title_draw.draw()
            item_list = [i for i in game_config.config_item if i not in character_data.item]
            handle_panel.text_list = item_list
            handle_panel.update()
            handle_panel.draw()
            return_list.extend(handle_panel.return_list)
            line = draw.LineDraw("-", self.width)
            line.draw()
            back_draw = draw.CenterButton(_("[返回]"), _("返回"), window_width)
            back_draw.draw()
            return_list.append(back_draw.return_text)
            yrn = flow_handle.askfor_all(return_list)
            if yrn == back_draw.return_text:
                cache.now_panel_id = constant.Panel.IN_SCENE
                break


class BuyItemByItemNameDraw:
    """
    点击后可购买道具的道具名字按钮对象
    Keyword arguments:
    text -- 道具id
    width -- 最大宽度
    is_button -- 绘制按钮
    num_button -- 绘制数字按钮
    button_id -- 数字按钮id
    """

    def __init__(self, text: int, width: int, is_button: bool, num_button: bool, button_id: int):
        """初始化绘制对象"""
        self.text = text
        """ 道具id """
        self.draw_text: str = ""
        """ 道具名字绘制文本 """
        self.width: int = width
        """ 最大宽度 """
        self.num_button: bool = num_button
        """ 绘制数字按钮 """
        self.button_id: int = button_id
        """ 数字按钮的id """
        self.button_return: str = str(button_id)
        """ 按钮返回值 """
        name_draw = draw.NormalDraw()
        item_config = game_config.config_item[self.text]
        if is_button:
            if num_button:
                index_text = text_handle.id_index(button_id)
                button_text = f"{index_text}{item_config.name}"
                name_draw = draw.LeftButton(
                    button_text, self.button_return, self.width, cmd_func=self.buy_item
                )
            else:
                button_text = f"[{item_config.name}]"
                name_draw = draw.CenterButton(
                    button_text, item_config.name, self.width, cmd_func=self.buy_item
                )
                self.button_return = item_config.name
            self.draw_text = button_text
        else:
            name_draw = draw.CenterDraw()
            name_draw.text = f"[{item_config.name}]"
            name_draw.width = self.width
            self.draw_text = name_draw.text
        self.now_draw = name_draw
        """ 绘制的对象 """

    def draw(self):
        """绘制对象"""
        self.now_draw.draw()

    def buy_item(self):
        py_cmd.clr_cmd()
        character_data: game_type.Character = cache.character_data[0]
        character_data.item.add(self.text)
        item_config = game_config.config_item[self.text]
        now_text = _("{nickname}购买了{item_name}").format(
            nickname=character_data.nick_name, item_name=item_config.name
        )
        now_draw = draw.WaitDraw()
        now_draw.text = now_text
        now_draw.width = window_width
        now_draw.draw()
        line_feed.draw()
        line_feed.draw()
