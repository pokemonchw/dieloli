import os
from types import FunctionType
from Script.Config import normal_config
from Script.UI.Model import panel, draw
from Script.UI.Panel import see_save_info_panel
from Script.Design import handle_panel, constant
from Script.Core import get_text, flow_handle, cache_control, game_type, py_cmd

config_normal = normal_config.config_normal
_: FunctionType = get_text._
""" 翻译api """
cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_panel.add_panel(constant.Panel.TITLE)
def title_panel():
    """绘制游戏标题菜单"""
    width = config_normal.text_width
    title_info = panel.TitleAndRightInfoListPanel()
    info_list = [config_normal.author, config_normal.verson_time, config_normal.verson]
    title_info.set(config_normal.game_name, info_list, width)
    title_info.draw()
    lineFeed = draw.NormalDraw()
    lineFeed.width = 1
    lineFeed.text = "\n"
    info = _("人类是可以被驯化的")
    lineFeed.draw()
    info_draw = draw.CenterDraw()
    info_draw.text = info
    info_draw.width = width
    info_draw.draw()
    lineFeed.draw()
    now_info_draw = draw.CenterDraw()
    now_info_draw.text = _("(注意这是一个临时的预览版本)")
    now_info_draw.width = width
    now_info_draw.draw()
    lineFeed.draw()
    lineFeed.draw()
    line = draw.LineDraw("=", width)
    line.draw()
    now_list = [_("开始游戏"), _("读取存档"), _("游戏设置"), _("成就列表")]
    button_panel = panel.OneMessageAndSingleColumnButton()
    button_panel.set(now_list, "", 0)
    button_panel.draw()
    return_list = button_panel.get_return_list()
    ans = flow_handle.askfor_all(return_list.keys())
    py_cmd.clr_cmd()
    now_key = return_list[ans]
    if now_key == now_list[0]:
        cache.now_panel_id = constant.Panel.CREATOR_CHARACTER
    elif now_key == now_list[1]:
        now_panel = see_save_info_panel.SeeSaveListPanel(width, 0)
        now_panel.draw()
    elif now_key == now_list[2]:
        cache.now_panel_id = constant.Panel.GAME_SETTING
    elif now_key == now_list[3]:
        cache.now_panel_id = constant.Panel.ACHIEVE
