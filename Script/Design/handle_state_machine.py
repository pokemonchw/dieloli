from functools import wraps
from types import FunctionType
from Script.Design import constant


def handle_state_machine(state_machine_id: int, character_id: int):
    """
    创建角色状态机
    Keyword arguments:
    state_machine_id -- 状态机id
    character_id -- 角色id
    """
    if state_machine_id in constant.handle_state_machine_data:
        constant.handle_state_machine_data[state_machine_id](character_id)


def add_state_machine(state_machine_id: int):
    """
    添加角色状态机函数
    Keyword arguments:
    state_machine_id -- 状态机id
    """

    def decorator(func: FunctionType):
        constant.handle_state_machine_data[state_machine_id] = func

    return decorator
