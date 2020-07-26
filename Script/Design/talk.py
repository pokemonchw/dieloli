import random
from functools import wraps
from Script.Core import cache_contorl, era_print


def add_talk(occupation: str, behavior_id: int) -> callable:
    """
    添加口上
    Keyword arguments:
    occupation -- 口上所属的职业
    behavior_id -- 口上对应的行为id
    Return arguments:
    callable -- 口上处理函数对象
    """

    def decorator(func):
        @wraps(func)
        def return_wrapper(*args, **kwargs):
            return random.choice(func(*args, **kwargs))

        cache_contorl.talk_data.setdefault(occupation, {})
        cache_contorl.talk_data[occupation][behavior_id] = return_wrapper
        return return_wrapper

    return decorator


def handle_talk(behavior_id: int):
    """
    处理行为结算对话
    Keyword arguments:
    behavior_id -- 行为id
    """
    now_target_character = cache_contorl.character_data[
        cache_contorl.character_data[0].target_character_id
    ]
    if (
        now_target_character.occupation in cache_contorl.talk_data
        and behavior_id
        in cache_contorl.talk_data[now_target_character.occupation]
    ):
        cache_contorl.talk_data[now_target_character.occupation][behavior_id]()
    else:
        era_print.multiple_line_return_print(
            cache_contorl.talk_data["default"][behavior_id]()
        )
    era_print.line_feed_print()
