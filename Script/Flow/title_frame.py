from Script.Panel import title_frame_panel
from Script.Core import py_cmd, cache_contorl
import os


def title_frame_func():
    """
    标题界面绘制流程
    """
    title_frame_panel.load_game_panel()
    cache_contorl.wframe_mouse["w_frame_re_print"] = 1
    ans = title_frame_panel.game_main_panel()
    py_cmd.clr_cmd()
    if ans == 0:
        cache_contorl.now_flow_id = "creator_character"
    elif ans == 1:
        cache_contorl.old_flow_id = "title_frame"
        cache_contorl.now_flow_id = "load_save"
    elif ans == 2:
        os._exit(0)
