import time
from Script.Core import (
    era_print,
    text_loading,
    text_handle,
    game_config,
    constant,
    py_cmd,
    cache_contorl,
)
from Script.Design import cmd_button_queue


def load_game_panel():
    """
    载入游戏动画绘制
    """
    era_print.next_screen_print()
    era_print.next_screen_print()
    era_print.one_by_one_print(1 / 3, text_loading.get_text_data(constant.FilePath.MESSAGE_PATH, "1"))
    era_print.line_feed_print()


def game_main_panel() -> int:
    """
    游戏标题界面主面板
    """
    era_print.restart_line_print()
    era_print.line_feed_print(text_handle.align(game_config.game_name, "center"))
    era_print.line_feed_print(text_handle.align(game_config.author, "right"))
    era_print.line_feed_print(text_handle.align(game_config.verson, "right"))
    era_print.line_feed_print(text_handle.align(game_config.verson_time, "right"))
    era_print.line_feed_print()
    era_print.restart_line_print()
    era_print.lines_center_print(1 / 3, text_loading.get_text_data(constant.FilePath.MESSAGE_PATH, "2"))
    time.sleep(1)
    era_print.line_feed_print()
    era_print.restart_line_print()
    time.sleep(1)
    py_cmd.focus_cmd()
    menu_int = cmd_button_queue.option_int(constant.CmdMenu.LOGO_MENU)
    return menu_int
