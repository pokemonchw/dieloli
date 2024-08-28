from typing import List
from types import FunctionType
from Script.UI.Moudle import panel, draw
from Script.Core import cache_control, get_text, text_handle, game_type, flow_handle, py_cmd
from Script.Config import game_config, config_def
from Script.Design import constant

_: FunctionType = get_text._
""" 翻译api """
line_feed = draw.NormalDraw()
""" 换行绘制对象 """
line_feed.text = "\n"
line_feed.width = 1
cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


class AchievePanel:
    """
    查看成就列表面板
    width -- 绘制宽度
    """

    def __init__(self, width: int):
        """初始化绘制对象"""
        self.width: int = width
        """ 最大绘制宽度 """
        self.return_list: List[str] = []
        """ 当前面板监听的按钮列表 """
        now_list = [cid for cid in game_config.config_achieve]
        self.handle_panel = panel.PageHandlePanel(
            now_list, AchieveInfoDraw, 10, 2, self.width, True, True
        )

    def draw(self):
        """绘制对象"""
        while 1:
            line_feed.draw()
            title_draw = draw.TitleLineDraw(_("成就列表"), self.width)
            title_draw.draw()
            self.return_list = []
            now_line = draw.LineDraw(".", self.width)
            self.handle_panel.update()
            self.handle_panel.draw()
            self.return_list.extend(self.handle_panel.return_list)
            now_line.draw()
            back_draw = draw.CenterButton(_("[返回]"), _("返回"), self.width)
            back_draw.draw()
            line_feed.draw()
            self.return_list.append(back_draw.return_text)
            yrn = flow_handle.askfor_all(self.return_list)
            py_cmd.clr_cmd()
            if yrn == back_draw.return_text:
                cache.now_panel_id = constant.Panel.TITLE
                break


class AchieveInfoDraw:
    """
    绘制成就信息按钮
    Keyword arguments:
    text -- 成就id
    width -- 最大宽度
    is_button -- 绘制按钮啊
    num_button -- 绘制数字按钮
    button_id -- 数字按钮的id
    """

    def __init__(self, text: str, width: int, is_button: bool, num_button: bool, button_id: int):
        """初始化绘制对象"""
        self.text = text
        """ 成就id """
        self.draw_text: str = ""
        """ 成就信息绘制文本 """
        self.width: int = width
        """ 最大宽度 """
        self.is_button: bool = is_button
        """ 绘制按钮 """
        self.is_num_button: bool = num_button
        """ 绘制数字按钮 """
        self.button_id: int = button_id
        """ 数字按钮的id """
        self.button_return: str = str(button_id)
        """ 按钮的返回值 """
        self.now_config: config_def.Achieve = game_config.config_achieve[self.text]
        """ 成就配置数据 """
        achieve_name = "??????????"
        if self.text in cache_control.achieve.completed_data and cache_control.achieve.completed_data[self.text]:
            achieve_name = self.now_config.name
        index_text = text_handle.id_index(self.button_id)
        self.draw_text = f"{index_text}{achieve_name}"
        self.draw_text = text_handle.align(self.draw_text,text_width=self.width)
        self.button_return = str(button_id)

    def draw(self):
        """绘制对象"""
        if (
            self.text not in cache_control.achieve.completed_data
            or
            not cache_control.achieve.completed_data[self.text]
        ) and self.now_config.hide:
            now_draw = draw.LeftButton(
                self.draw_text, self.button_return, cmd_func=self.draw_hide_info, width=self.width
            )
        else:
            now_draw = draw.LeftButton(
                self.draw_text, self.button_return, cmd_func=self.draw_info, width=self.width
            )
        now_draw.draw()

    def draw_hide_info(self):
        """提示成就解锁条件被隐藏"""
        now_draw = draw.WaitDraw()
        now_draw.text = _("成就的解锁条件被偷偷藏起来了~") + "\n"
        now_draw.width = self.width
        now_draw.draw()

    def draw_info(self):
        """提示成就的解锁条件"""
        now_draw = draw.WaitDraw()
        now_draw.text = self.now_config.info + "\n"
        now_draw.width = self.width
        now_draw.draw()
