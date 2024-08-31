import random
from Script.Core import cache_control, game_type
from Script.Design import game_time
from Script.Config import game_config, config_def, normal_config
from Script.UI.Moudle import draw


cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


def handle_weather(draw_info:bool = True):
    """
    处理天气
    Keyword arguments:
    draw_info -- 是否绘制天气信息
    """
    now_solar = game_time.get_solar_period(cache.game_time)
    now_weather = game_config.config_solar_period_weather_data[now_solar][0]
    if len(game_config.config_solar_period_weather_data[now_solar]) > 1:
        now_weather = random.choice(game_config.config_solar_period_weather_data[now_solar])
    old_weather = cache.weather
    cache.weather = now_weather
    weather_config: config_def.Weather = game_config.config_weather[now_weather]
    cache.weather_last_time = random.randint(weather_config.min_time, weather_config.max_time)
    if draw_info:
        if old_weather != now_weather:
            now_draw = draw.WaitDraw()
            now_draw.text = "\n" + weather_config.info + "\n"
            now_draw.width = normal_config.config_normal.text_width
            now_draw.draw()
