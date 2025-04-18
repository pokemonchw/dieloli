from typing import Set, Tuple, Dict
from types import FunctionType
from uuid import UUID
import time
import ollama
from Script.Core import get_text, game_type, cache_control, flow_handle, text_handle, py_cmd
from Script.UI.Model import panel, draw
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
        panel_type_list = [_("语言"),_("NSFW"),_("AI")]
        return_list = []
        for panel_type in panel_type_list:
            if self.now_panel == panel_type:
                now_draw = draw.CenterButton(
                    f"[{panel_type}]",
                    panel_type,
                    8,
                    " ",
                    "onbutton",
                    "standard",
                    cmd_func=self.change_panel,
                    args=(panel_type,),
                )
            else:
                now_draw = draw.CenterButton(
                    f"[{panel_type}]",
                    panel_type,
                    8,
                    cmd_func=self.change_panel,
                    args=(panel_type,),
                )
            return_list.append(now_draw.return_text)
            now_draw.draw()
        line_feed.draw()
        title_line = draw.LineDraw("-.-", self.width)
        title_line.draw()
        language_option = []
        ai_mode_option = {}
        if self.now_panel == _("语言"):
            language_panel = SystemLanguageSettingPanel(self.width)
            language_panel.draw()
            language_option = language_panel.return_list
            return_list.extend(language_option.keys())
        elif self.now_panel == _("NSFW"):
            nsfw_panel = SystemNSFWSettingPanel(self.width)
            nsfw_panel.draw()
            return_list.extend(nsfw_panel.return_list)
        elif self.now_panel == ("AI"):
            ai_panel = SystemAISettingPanel(self.width)
            ai_panel.draw()
            return_list.extend(ai_panel.return_list.keys())
            ai_mode_option = ai_panel.return_list
        back_draw = draw.CenterButton(_("[返回]"),_("返回"),self.width)
        back_draw.draw()
        line_feed.draw()
        return_list.append(back_draw.return_text)
        yrn = flow_handle.askfor_all(return_list)
        if yrn == back_draw.return_text:
            cache.now_panel_id = constant.Panel.TITLE
        elif self.now_panel == _("语言"):
            if yrn in language_option:
                choice_language = language_option[yrn]
                language_id = game_config.config_system_language_data[choice_language]
                normal_config.change_normal_config("language",language_id)
                now_draw = draw.WaitDraw()
                now_draw.text = _("请重启游戏以让设置生效")
                now_draw.width = self.width
                now_draw.draw()
        elif self.now_panel == _("NSFW"):
            if yrn == "0":
                if normal_config.config_normal.nsfw:
                    normal_config.change_normal_config("nsfw", 0)
                    normal_config.config_normal.nsfw = 0
                else:
                    normal_config.change_normal_config("nsfw", 1)
                    normal_config.config_normal.nsfw = 1
        elif self.now_panel == _("AI"):
            if yrn in ai_mode_option:
                if ai_mode_option[yrn] == _("关闭AI模式"):
                    normal_config.change_normal_config("ai_mode", 0)
                elif ai_mode_option[yrn] == _("启用ollama模式"):
                    try:
                        models = ollama.list()
                        try:
                            model_info = ollama.show(normal_config.config_normal.ollama_mode)
                            normal_config.change_normal_config("ai_mode", 1)
                        except Exception as e:
                            now_draw = draw.LeftDraw()
                            now_draw.text = _("模型未安装，正在自动安装，请稍等(需联网)")
                            now_draw.width = self.width
                            now_draw.draw()
                            install_result = ollama.pull(normal_config.config_normal.ollama_mode)
                            model_installed = False
                            for i in range(60):
                                try:
                                    model_info = ollama.show(normal_config.config_normal.ollama_mode)
                                    model_installed = True
                                    break
                                except Exception as e:
                                    time.sleep(2)
                            if model_installed:
                                normal_config.change_normal_config("ai_mode", 1)
                            else:
                                now_draw = draw.LeftDraw()
                                now_draw.text = _("模型安装失败，已关闭ollama模式，请检查设置")
                                now_draw.width = self.width
                                now_draw.draw()
                    except Exception as e:
                        now_draw = draw.WaitDraw()
                        now_draw.text = _("无法链接ollama服务，请确保ollama服务正常启动")
                        now_draw.width = self.width
                        now_draw.draw()
                        normal_config.change_normal_config("ai_mode", 0)
                elif ai_mode_option[yrn] == _("启用外部api模式"):
                    normal_config.change_normal_config("ai_mode", 2)
        py_cmd.clr_cmd()

    def change_panel(self, panel_type: str):
        """
        切换当前绘制的面板
        Keyword arguments:
        now_type -- 切换的面板id
        """
        self.now_panel = panel_type
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
        now_language = game_config.config_system_language[normal_config.config_normal.language].name
        now_panel.set(language_list,_("请选择需要切换的语言，当前语言:{now_language}").format(now_language=now_language))
        now_panel.draw()
        self.return_list = now_panel.get_return_list()


class SystemNSFWSettingPanel:
    """
    用于设置是否开启NSFW内容的开关的面板对象
    Keyword arguments:
    width -- 绘制宽度
    """

    def __init__(self, width: int):
        """ 初始化绘制对象 """
        self.width: int = width
        """ 绘制的最大宽度 """

    def draw(self):
        """ 绘制对象 """
        title_line = draw.LineDraw("o", self.width)
        now_panel = panel.OneMessageAndSingleColumnButton()
        ask_for_list = [_("切换")]
        if normal_config.config_normal.nsfw:
            now_panel.set(ask_for_list, _("是否关闭NSFW内容"))
        else:
            now_panel.set(ask_for_list, _("是否开启NSFW内容"))
        now_panel.draw()
        self.return_list = now_panel.get_return_list()


class SystemAISettingPanel:
    """
    用于设置是否开启AI模式的面板对象
    Keyword arguments:
    width -- 绘制宽度
    """

    def __init__(self, width: int):
        """ 初始化绘制对象 """
        self.width: int = width
        """ 绘制的最大宽度 """

    def draw(self):
        """ 绘制对象 """
        title_line = draw.LineDraw("o", self.width)
        now_panel = panel.OneMessageAndSingleColumnButton()
        info_text = _("是否开启AI模式?\n请注意:\nollama模式需本地安装ollama，同时对电脑配置要求较高\n外部api模式需要联网进行游戏，同时本游戏对tokens消耗较高\n另外目前AI模式为实验性模式，产出内容无法良好控制")
        ask_for_list = []
        if normal_config.config_normal.ai_mode == 0:
            ask_for_list = [_("启用ollama模式"),_("启用外部api模式")]
        elif normal_config.config_normal.ai_mode == 1:
            ask_for_list = [_("关闭AI模式"),_("启用外部api模式")]
        elif normal_config.config_normal.ai_mode == 2:
            ask_for_list = [_("关闭AI模式"),_("启用ollama模式")]
        now_panel.set(ask_for_list, info_text)
        now_panel.draw()
        self.return_list = now_panel.get_return_list()
