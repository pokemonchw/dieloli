from functools import wraps
from Script.Core import cache_contorl, constant
from Script.Design import game_time, talk, map_handle, talk_cache


def handle_settle_behavior(character_id: int):
    """
    处理结算角色行为
    Keyword arguments:
    character_id -- 角色id
    """
    cache_contorl.settle_behavior_data[cache_contorl.character_data[character_id].behavior["BehaviorId"]](
        character_id
    )


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

        cache_contorl.settle_behavior_data[behavior_id] = return_wrapper
        return return_wrapper

    return decorator
