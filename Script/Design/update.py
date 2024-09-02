import time
import datetime
from Script.Design import character_behavior, game_time, event, handle_achieve, weather
from Script.Core import py_cmd,cache_control, game_type
from Script.UI.Moudle import draw
from Script.Config import normal_config

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


def game_update_flow(add_time: int):
    """
    游戏时间步进
    Keyword arguments:
    add_time -- 游戏步进的时间
    """
    py_cmd.focus_cmd()
    event.handle_event(0, 1, cache.game_time, cache.game_time)
    line_feed_draw = draw.NormalDraw()
    line_feed_draw.text = "\n"
    now_time = cache.game_time
    next_hour = now_time + 3600
    next_hour_date_time = datetime.datetime.fromtimestamp(next_hour)
    fix_next_hour_data_time = datetime.datetime(next_hour_date_time.year, next_hour_date_time.month, next_hour_date_time.day, next_hour_date_time.hour)
    next_hour_time = fix_next_hour_data_time.timestamp()
    fix_next_hour_time = (next_hour_time - now_time) / 60
    cache.character_data[0].behavior.start_time = cache.game_time
    for _ in range(add_time):
        cache.game_time += 60
        if cache.game_time == next_hour_time:
            if fix_next_hour_time > 30:
                fix_next_hour_time = 30
            character_behavior.init_character_behavior()
            time_draw = draw.CenterDraw()
            time_draw.text = game_time.get_date_text(cache.game_time)
            time_draw.width = normal_config.config_normal.text_width
            time_draw.draw()
            line_feed_draw.draw()
            now_time = cache.game_time
            next_hour = now_time + 3600
            next_hour_date_time = datetime.datetime.fromtimestamp(next_hour)
            fix_next_hour_data_time = datetime.datetime(next_hour_date_time.year, next_hour_date_time.month, next_hour_date_time.day, next_hour_date_time.hour)
            next_hour_time = fix_next_hour_data_time.timestamp()
            fix_next_hour_time = (next_hour_time - now_time) / 60
        character_behavior.init_character_behavior()
        cache.weather_last_time -= 1
        if cache.weather_last_time <= 0:
            weather.handle_weather()
    handle_achieve.check_all_achieve()

