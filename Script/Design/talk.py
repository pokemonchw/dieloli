import random
from functools import wraps
from Script.Core import cache_contorl, era_print
from Script.Design import map_handle


def add_talk(behavior_id: int, talk_id: int, premise_list: set) -> callable:
    """
    添加口上
    Keyword arguments:
    behavior_id -- 口上对应的行为id
    talk_id -- 口上id
    premise_list -- 口上前提集合
    Return arguments:
    callable -- 口上处理函数对象
    """

    def decorator(func):
        @wraps(func)
        def return_wrapper(*args, **kwargs):
            return random.choice(func(*args, **kwargs))

        cache_contorl.talk_data.setdefault(behavior_id, {})
        cache_contorl.talk_data[behavior_id][talk_id] = return_wrapper
        cache_contorl.premise_talk_table.setdefault(behavior_id, {})
        cache_contorl.premise_talk_table[behavior_id][talk_id] = premise_list
        return return_wrapper

    return decorator


def handle_talk(character_id: int):
    """
    处理行为结算对话
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache_contorl.character_data[character_id]
    behavior_id = character_data.behavior["BehaviorId"]
    era_print.line_feed_print()
    now_talk_data = {}
    if behavior_id in cache_contorl.talk_data:
        for talk_id in cache_contorl.talk_data[behavior_id]:
            now_weight = 1
            for premise in cache_contorl.premise_talk_table[behavior_id][
                talk_id
            ]:
                now_add_weight = cache_contorl.handle_premise_data[premise](
                    character_id
                )
                if now_add_weight:
                    now_weight += now_add_weight
                else:
                    now_weight = 0
                    break
            if now_weight:
                now_talk_data.setdefault(now_weight, set())
                now_talk_data[now_weight].add(talk_id)
    if len(now_talk_data):
        max_weight = max(list(now_talk_data.keys()))
        now_talk_id = random.choice(list(now_talk_data[max_weight]))
        now_talk: str = cache_contorl.talk_data[behavior_id][now_talk_id]()
        scene_path = cache_contorl.character_data[0].position
        scene_path_str = map_handle.get_map_system_path_str_for_list(
            scene_path
        )
        scene_data = cache_contorl.scene_data[scene_path_str]
        scene_name = scene_data["SceneName"]
        player_data = cache_contorl.character_data[0]
        target_data = cache_contorl.character_data[
            character_data.target_character_id
        ]
        now_talk = now_talk.format(
            NickName=character_data.nick_name,
            FoodName=character_data.behavior["FoodName"],
            Name=character_data.name,
            SceneName=scene_name,
            PlayerNickName=player_data.nick_name,
            TargetName=target_data.name,
        )
        era_print.multiple_line_return_print(now_talk)
        era_print.line_feed_print()
