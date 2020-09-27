# -*- coding: UTF-8 -*-
from Script.Core import py_cmd, cache_contorl


def start_frame():
    """
    游戏主流程
    """
    cache_contorl.now_panel_id = "title"
    while True:
        py_cmd.clr_cmd()
        cache_contorl.panel_data[cache_contorl.now_panel_id]()
