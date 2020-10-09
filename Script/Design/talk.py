import random
from functools import wraps
from types import FunctionType
from Script.Core import cache_contorl, era_print
from Script.Design import map_handle


def add_talk(behavior_id: int, talk_id: int, premise_list: set, adv=0) -> FunctionType:
    """
    添加口上
    Keyword arguments:
    behavior_id -- 口上对应的行为id
    talk_id -- 口上id
    premise_list -- 口上前提集合
    adv -- advid,不为0时为id对应自定义角色限定口上
    Return arguments:
    FunctionType -- 口上处理函数对象
    """

    def decorator(func):
        @wraps(func)
        def return_wrapper(*args, **kwargs):
            return random.choice(func(*args, **kwargs))

        cache_contorl.adv_talk_data.setdefault(adv, {})
        cache_contorl.adv_talk_data[adv].setdefault(behavior_id, {})
        cache_contorl.adv_talk_data[adv][behavior_id][talk_id] = return_wrapper
        cache_contorl.premise_talk_table.setdefault(adv, {})
        cache_contorl.premise_talk_table[adv].setdefault(behavior_id, {})
        cache_contorl.premise_talk_table[adv][behavior_id][talk_id] = premise_list
        return return_wrapper

    return decorator


def handle_talk(character_id):
    """
    处理行为结算对话
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache_contorl.character_data[character_id]
    behavior_id = character_data.behavior["BehaviorId"]
    era_print.line_feed_print()
    now_talk_data = {}
    if behavior_id in cache_contorl.adv_talk_data[0]:
        for talk_id in cache_contorl.adv_talk_data[0][behavior_id]:
            now_weight = 1
            for premise in cache_contorl.premise_talk_table[0][behavior_id][talk_id]:
                now_add_weight = cache_contorl.handle_premise_data[premise](character_id)
                if now_add_weight:
                    now_weight += now_add_weight
                else:
                    now_weight = 0
                    break
            if now_weight:
                now_talk_data.setdefault(now_weight, set())
                now_talk_data[now_weight].add(talk_id)
    now_adv_talk = {}
    if character_data.adv:
        adv_talk_data = cache_contorl.adv_talk_data[character_data.adv]
        if behavior_id in adv_talk_data:
            for talk_id in adv_talk_data[behavior_id]:
                now_weight = 1
                for premise in cache_contorl.premise_talk_table[character_data.adv][behavior_id][talk_id]:
                    now_add_weight = cache_contorl.handle_premise_data[premise](character_id)
                    if now_add_weight:
                        now_weight += now_add_weight
                    else:
                        now_weight = 0
                        break
                if now_weight:
                    now_adv_talk.setdefault(now_weight, set())
                    now_adv_talk[now_weight].add(talk_id)
    now_talk = None
    adv_weight = 0
    talk_weight = 0
    if len(now_adv_talk):
        adv_weight = max(list(now_adv_talk.keys()))
        now_talk_id = random.choice(list(now_adv_talk[adv_weight]))
        now_talk = cache_contorl.adv_talk_data[character_data.adv][behavior_id][now_talk_id]
    if len(now_talk_data):
        talk_weight = max(list(now_talk_data.keys()))
        if talk_weight > adv_weight:
            now_talk_id = random.choice(list(now_talk_data[talk_weight]))
            now_talk = cache_contorl.adv_talk_data[0][behavior_id][now_talk_id]
    if now_talk != None:
        now_talk_text: str = now_talk()
        scene_path = cache_contorl.character_data[0].position
        scene_path_str = map_handle.get_map_system_path_str_for_list(scene_path)
        scene_data = cache_contorl.scene_data[scene_path_str]
        scene_name = scene_data["SceneName"]
        player_data = cache_contorl.character_data[0]
        target_data = cache_contorl.character_data[character_data.target_character_id]
        now_talk_text = now_talk_text.format(
            NickName=character_data.nick_name,
            FoodName=character_data.behavior["FoodName"],
            Name=character_data.name,
            SceneName=scene_name,
            PlayerNickName=player_data.nick_name,
            TargetName=target_data.name,
        )
        era_print.multiple_line_return_print(now_talk_text)
        era_print.line_feed_print()