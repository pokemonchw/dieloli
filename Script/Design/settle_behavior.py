from functools import wraps
from Script.Core import cache_control, constant, game_type
from Script.Design import game_time, talk, map_handle

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


def handle_settle_behavior(character_id: int):
    """
    处理结算角色行为
    Keyword arguments:
    character_id -- 角色id
    """
    cache.settle_behavior_data[cache.character_data[character_id].behavior.behavior_id](character_id)


def add_settle_behavior(behavior_id: int):
    """
    添加行为结算处理
    Keyword arguments:
    behavior_id -- 行为id
    """

    def decorator(func):
        @wraps(func)
        def return_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        cache.settle_behavior_data[behavior_id] = return_wrapper
        return return_wrapper

    return decorator
