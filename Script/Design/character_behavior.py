import os
import random
from functools import wraps
from Script.Core import (
    cache_contorl,
    game_path_config,
    game_type,
    constant,
)
from Script.Design import (
    settle_behavior,
    game_time,
    character,
    handle_premise,
    talk,
)

game_path = game_path_config.game_path


def init_character_behavior():
    """
    角色行为树总控制
    """
    while 1:
        if len(cache_contorl.over_behavior_character) >= len(cache_contorl.character_data):
            break
        for character_id in cache_contorl.character_data:
            character_behavior(character_id)
    cache_contorl.over_behavior_character = {}


def character_behavior(character_id: int):
    """
    角色行为控制
    Keyword arguments:
    character_id -- 角色id
    """
    if character_id in cache_contorl.over_behavior_character:
        return
    if cache_contorl.character_data[character_id].behavior["StartTime"] == None:
        character.init_character_behavior_start_time(character_id)
    game_time.init_now_course_time_slice(character_id)
    if cache_contorl.character_data[character_id].state == constant.CharacterStatus.STATUS_ARDER:
        if character_id:
            character_target_judge(character_id)
        else:
            cache_contorl.over_behavior_character[0] = 1
    else:
        status_judge = judge_character_status(character_id)
        if status_judge:
            cache_contorl.over_behavior_character[character_id] = 1


def character_target_judge(character_id: int):
    """
    查询角色可用目标活动并执行
    Keyword arguments:
    character_id -- 角色id
    """
    target, _, judge = search_target(character_id, list(cache_contorl.handle_target_data.keys()), set())
    if judge:
        cache_contorl.handle_target_data[target](character_id)
    else:
        start_time = cache_contorl.character_data[character_id].behavior["StartTime"]
        now_judge = game_time.judge_date_big_or_small(start_time, cache_contorl.game_time)
        if now_judge:
            cache_contorl.over_behavior_character[character_id] = 1
        else:
            next_time = game_time.get_sub_date(minute=1, old_date=start_time)
            cache_contorl.character_data[character_id].behavior["StartTime"] = next_time


def judge_character_status(character_id: int) -> int:
    """
    校验并结算角色状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    bool -- 本次update时间切片内活动是否已完成
    """
    character_data = cache_contorl.character_data[character_id]
    start_time = character_data.behavior["StartTime"]
    end_time = game_time.get_sub_date(minute=character_data.behavior["Duration"], old_date=start_time)
    now_time = cache_contorl.game_time
    time_judge = game_time.judge_date_big_or_small(now_time, end_time)
    add_time = (end_time.timestamp() - start_time.timestamp()) / 60
    character_data.status["BodyFeeling"]["Hunger"] += add_time * 0.02
    character_data.status["BodyFeeling"]["Thirsty"] += add_time * 0.02
    if time_judge:
        settle_behavior.handle_settle_behavior(character_id)
        talk.handle_talk(character_id)
        character_data.behavior["BehaviorId"] = constant.Behavior.SHARE_BLANKLY
        character_data.state = constant.CharacterStatus.STATUS_ARDER
    if time_judge == 1:
        character_data.behavior["StartTime"] = end_time
        return 0
    elif time_judge == 2:
        character.init_character_behavior_start_time(character_id)
        return 0
    return 1


def search_target(character_id: int, target_list: list, null_target: set) -> (str, int, bool):
    """
    查找可用目标
    Keyword arguments:
    character_id -- 角色id
    target_list -- 检索的目标列表
    null_target -- 被排除的目标
    Return arguments:
    目标id
    int -- 目标权重
    bool -- 前提是否能够被满足
    """
    target_data = {}
    for target in target_list:
        if target in null_target:
            continue
        target_premise_list = cache_contorl.premise_target_table[target]
        now_weiget = 0
        now_target_pass_judge = 0
        now_target_data = {}
        premise_judge = 1
        for premise in target_premise_list:
            premise_judge = handle_premise.handle_premise(premise, character_id)
            if premise_judge > 0:
                now_weiget += premise_judge
            else:
                premise_judge = 0
                if premise in cache_contorl.effect_target_table:
                    now_target_list = cache_contorl.effect_target_table[premise]
                    now_target, now_target_weight, now_judge = search_target(
                        character_id, now_target_list, null_target
                    )
                    if now_judge:
                        now_target_data.setdefault(now_target_weight, set())
                        now_target_data[now_target_weight].add(now_target)
                    else:
                        now_target_pass_judge = 1
                        break
                else:
                    now_target_pass_judge = 1
                    break
        if now_target_pass_judge:
            null_target.add(target)
            continue
        if premise_judge:
            target_data.setdefault(now_weiget, set())
            target_data[now_weiget].add(target)
        else:
            now_max_weight = max(list(now_target_data))
            target_data.setdefault(now_max_weight, set())
            target_data[now_max_weight].add(random.choice(list(now_target_data[now_max_weight])))
    if len(target_data):
        max_weight = max(list(target_data.keys()))
        return random.choice(list(target_data[max_weight])), max_weight, 1
    return "", 0, 0
