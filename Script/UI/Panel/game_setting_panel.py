from typing import Set, Tuple, Dict
from types import FunctionType
from uuid import UUID
from Script.Core import get_text, game_type, cache_control, flow_handle, text_handle, py_cmd
from Script.UI.Moudle import panel, draw
from Script.Design import cooking, update, constant
from Script.Config import normal_config, game_config

_: FunctionType = get_text._
""" 翻译api """
window_width: int = normal_config.config_normal.text_width
""" 窗体宽度 """
cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
line_feed = draw.NormalDraw()
""" 换行绘制对象 """
line_feed.text = "\n"
line_feed.width = 1


class GameSettingPanel:
    """
    用于设置游戏的面板对象
    Keyword arguments:
    width -- 绘制宽度
    """

    def __init__(self, width: int):
        """初始化绘制对象"""
        self.width: int = width
        """ 绘制的最大宽度 """
        self.now_panel = _("语言")
        """ 当前的面板 """

    def draw(self):
        """绘制对象"""
        title_draw = draw.TitleLineDraw(_("游戏设置"),self.width)
        title_draw.draw()
        panel_type_list = [_("语言")]
        now_draw = draw.CenterDraw()
        now_draw.text = panel_type_list[0]
        now_draw.style = "onbutton"
        now_draw.draw()
        line_feed.draw()
        title_line = draw.LineDraw("-.-", self.width)
        title_line.draw()
        return_list = []
        if self.now_panel == _("语言"):
            language_panel = SystemLanguageSettingPanel(self.width)
            language_panel.draw()
            language_option = language_panel.return_list
            return_list.extend(language_option.keys())
        back_draw = draw.CenterButton(_("[返回]"),_("返回"),self.width)
        back_draw.draw()
        return_list.append(back_draw.return_text)
        yrn = flow_handle.askfor_all(return_list)
        if yrn == back_draw.return_text:
            cache.now_panel_id = constant.Panel.TITLE
        elif self.now_panel == _("语言"):
            if yrn in language_option:
                choice_language = language_option[yrn]
                language_id = game_config.config_system_language_data[choice_language]
                normal_config.change_normal_config("language",language_id)
                now_draw = draw.LeftDraw()
                now_draw.text = _("请重启游戏以让设置生效")
                now_draw.width = self.width
                now_draw.draw()
        py_cmd.clr_cmd()



class SystemLanguageSettingPanel:
    """
    用于设置游戏语言的面板对象
    Keyword arguments:
    width -- 绘制宽度
    """

    def __init__(self, width: int):
        """初始化绘制对象"""
        self.width: int = width
        """ 绘制的最大宽度 """
        self.return_list: Dict[str,str] = []
        """ 面板监听的返回列表 按钮id:语言id """

    def draw(self):
        """绘制对象"""
        title_line = draw.LineDraw("o",self.width)
        now_panel = panel.OneMessageAndSingleColumnButton()
        language_list = [f"{i.name}" for i in game_config.config_system_language.values()]
        now_panel.set(language_list,_("请选择需要切换的语言"))
        now_panel.draw()
        self.return_list = now_panel.get_return_list()
