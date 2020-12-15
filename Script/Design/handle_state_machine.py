from functools import wraps
from types import FunctionType
from Script.Core import cache_control, game_type


cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


def handle_state_machine(state_machine_id: int, character_id: int):
    """
    创建角色状态机
    Keyword arguments:
    state_machine_id -- 状态机id
    character_id -- 角色id
    """
    if state_machine_id in cache.handle_state_machine_data:
        cache.handle_state_machine_data[state_machine_id](character_id)


def add_state_machine(state_machine_id: int):
    """
    添加角色状态机函数
    Keyword arguments:
    state_machine_id -- 状态机id
    """

    def decorator(func: FunctionType):
        @wraps(func)
        def return_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        cache.handle_state_machine_data[state_machine_id] = return_wrapper
        return return_wrapper

    return decorator
