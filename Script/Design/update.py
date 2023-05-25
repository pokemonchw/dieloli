import time
import datetime
from Script.Design import character_behavior, game_time, event
from Script.Core import py_cmd,cache_control, game_type
from Script.UI.Moudle import draw
from Script.Config import normal_config

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


def game_update_flow(add_time: int):
    """
    游戏流程刷新
    Keyword arguments:
    add_time -- 游戏步进的时间
    """
    now_event = event.handle_event(0,1)
    if now_event != None:
        now_event.draw()
    line_feed_draw = draw.NormalDraw()
    line_feed_draw.text = "\n"
    while 1:
        now_time = cache.game_time
        now_date_time = datetime.datetime.fromtimestamp(now_time)
        next_hour = now_time + 3600
        next_hour_date_time = datetime.datetime.fromtimestamp(next_hour)
        fix_next_hour_data_time = datetime.datetime(next_hour_date_time.year, next_hour_date_time.month, next_hour_date_time.day, next_hour_date_time.hour)
        next_hour_time = fix_next_hour_data_time.timestamp()
        fix_next_hour_time = (next_hour_time - now_time) / 60
        if fix_next_hour_time and fix_next_hour_time < add_time:
            if fix_next_hour_time > 30:
                fix_next_hour_time = 30
            game_time.sub_time_now(fix_next_hour_time)
            character_behavior.init_character_behavior()
            add_time -= fix_next_hour_time
            time_draw = draw.CenterDraw()
            time_draw.text = game_time.get_date_text(cache.game_time)
            time_draw.width = normal_config.config_normal.text_width
            time_draw.draw()
            line_feed_draw.draw()
        else:
            game_time.sub_time_now(add_time)
            character_behavior.init_character_behavior()
            time_draw = draw.CenterDraw()
            time_draw.text = game_time.get_date_text(cache.game_time)
            time_draw.width = normal_config.config_normal.text_width
            time_draw.draw()
            line_feed_draw.draw()
            break
    py_cmd.focus_cmd()
