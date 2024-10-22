import datetime
from typing import List
from types import FunctionType
from Script.Core import (
    cache_control,
    get_text,
    save_handle,
    text_handle,
    flow_handle,
    game_type,
    py_cmd,
)
from Script.Config import normal_config
from Script.UI.Moudle import panel, draw
from Script.Design import game_time, constant

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
_: FunctionType = get_text._
""" 翻译api """
window_width = normal_config.config_normal.text_width
""" 屏幕宽度 """
line_feed = draw.NormalDraw()
""" 换行绘制对象 """
line_feed.text = "\n"
line_feed.width = 1


class SeeSaveListPanel:
    """
    查看存档列表面板
    width -- 绘制宽度
    write_save -- 是否存储存档
    """

    def __init__(self, width: int, write_save: bool):
        """初始化绘制对象"""
        self.width: int = width
        """ 最大绘制宽度 """
        self.return_list: List[str] = []
        """ 当前面板监听的按钮列表 """
        now_list = [(i, write_save) for i in range(normal_config.config_normal.max_save)]
        self.handle_panel = panel.PageHandlePanel(
            now_list, SaveInfoDraw, normal_config.config_normal.save_page, 1, width, 1, 1, 0, "=-="
        )
        """ 页面控制对象 """

    def draw(self):
        """绘制对象"""
        while 1:
            if cache.back_save_panel:
                cache.back_save_panel = 0
                break
            line_feed.draw()
            title_draw = draw.TitleLineDraw(_("存档列表"), self.width)
            title_draw.draw()
            self.return_list = []
            auto_save_draw = SaveInfoDraw(["auto", 0], self.width, 1, 0, 0)
            auto_save_draw.draw()
            line_feed.draw()
            self.return_list.append("auto")
            now_line = draw.LineDraw(".", self.width)
            now_line.draw()
            self.handle_panel.update()
            self.handle_panel.draw()
            self.return_list.extend(self.handle_panel.return_list)
            back_draw = draw.CenterButton(_("[返回]"), _("返回"), self.width)
            back_draw.draw()
            line_feed.draw()
            self.return_list.append(back_draw.return_text)
            yrn = flow_handle.askfor_all(self.return_list)
            py_cmd.clr_cmd()
            if yrn == back_draw.return_text:
                break


class SaveInfoDraw:
    """
    绘制存档信息按钮
    Keyword arguments:
    text -- 存档id
    width -- 最大宽度
    is_button -- 绘制按钮
    num_button -- 绘制数字按钮
    button_id -- 数字按钮的id
    """

    def __init__(self, text: str, width: int, is_button: bool, num_button: bool, button_id: int):
        """初始化绘制对象"""
        self.text: str = str(text[0])
        """ 存档id """
        self.write_save: bool = text[1]
        """ 是否存储存档 """
        self.draw_text: str = ""
        """ 存档信息绘制文本 """
        self.width: int = width
        """ 最大宽度 """
        self.is_button: bool = is_button
        """ 绘制按钮 """
        self.is_num_button: bool = num_button
        """ 绘制数字按钮 """
        self.button_id: int = button_id
        """ 数字按钮的id """
        self.button_return: str = str(button_id)
        """ 按钮返回值 """
        self.save_exist_judge = save_handle.judge_save_file_exist(self.text)
        """ 存档位是否已存在 """
        save_name = _("空存档位")
        if self.save_exist_judge:
            save_head = save_handle.load_save_info_head(self.text)
            game_time: datetime.datetime = datetime.datetime.fromtimestamp(save_head["game_time"])
            save_time: datetime.datetime = save_head["save_time"]
            game_time_text = _("游戏时间:") + game_time.strftime("%Y-%m-%d %H:%M")
            save_time_text = _("存档时间:") + save_time.strftime("%Y-%m-%d %H:%M")
            save_name = f"No.{self.text} {save_head['game_verson']} {game_time_text} {save_head['character_name']} {save_time_text}"
        if is_button:
            if num_button:
                index_text = text_handle.id_index(button_id)
                now_text_width = self.width - len(index_text)
                new_text = text_handle.align(save_name, "center", text_width=now_text_width)
                self.draw_text = f"{index_text}{new_text}"
                self.button_return = str(button_id)
            else:
                new_text = text_handle.align(save_name, "center", text_width=self.width)
                self.draw_text = new_text
                self.button_return = text[0]
        else:
            new_text = text_handle.align(save_name, "center", text_width=self.width)
            self.draw_text = new_text

    def draw(self):
        """绘制对象"""
        if self.is_button and (self.save_exist_judge or self.write_save):
            now_draw = draw.Button(
                self.draw_text, self.button_return, cmd_func=self.draw_save_handle
            )
        else:
            now_draw = draw.NormalDraw()
            now_draw.text = self.draw_text
        now_draw.width = self.width
        now_draw.draw()

    def draw_save_handle(self):
        """处理读写存档"""
        py_cmd.clr_cmd()
        line_feed.draw()
        if self.save_exist_judge:
            now_ask_list = []
            if self.write_save:
                now_ask_list = [_("读取"), _("覆盖"), _("删除"), _("返回")]
            else:
                now_ask_list = [_("读取"), _("删除"), _("返回")]
            button_panel = panel.OneMessageAndSingleColumnButton()
            button_panel.set(now_ask_list, _("准备如何处理这个存档?"), 0)
            button_panel.draw()
            return_list = button_panel.get_return_list()
            ans = flow_handle.askfor_all(return_list.keys())
            py_cmd.clr_cmd()
            now_key = return_list[ans]
            if now_key == _("读取"):
                self.load_save()
            elif now_key == _("覆盖"):
                save_handle.establish_save(self.text)
            elif now_key == _("删除"):
                self.delete_save()
        else:
            save_handle.establish_save(self.text)

    def load_save(self):
        """载入存档"""
        save_handle.input_load_save(str(self.text))
        cache.now_panel_id = constant.Panel.IN_SCENE
        cache.back_save_panel = 1
        flow_handle.open_eventbox()

    def delete_save(self):
        """删除存档"""
        save_handle.remove_save(self.text)
