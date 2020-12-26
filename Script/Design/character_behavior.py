import os
import random
import datetime
import time
from functools import wraps
from Script.Core import (
    cache_control,
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
from Script.Config import game_config

game_path = game_path_config.game_path
cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


def init_character_behavior():
    """
    角色行为树总控制
    """
    while 1:
        if len(cache.over_behavior_character) >= len(cache.character_data):
            break
        for character_id in cache.character_data:
            character_behavior(character_id, cache.game_time)
    cache.over_behavior_character = {}


def character_behavior(character_id: int, now_time: datetime.datetime):
    """
    角色行为控制
    Keyword arguments:
    character_id -- 角色id
    now_time -- 指定时间
    """
    if character_id in cache.over_behavior_character:
        return
    if cache.character_data[character_id].behavior.start_time == None:
        character.init_character_behavior_start_time(character_id, now_time)
    game_time.init_now_course_time_slice(character_id)
    if cache.character_data[character_id].state == constant.CharacterStatus.STATUS_ARDER:
        if character_id:
            character_target_judge(character_id, now_time)
        else:
            cache.over_behavior_character[0] = 1
    else:
        status_judge = judge_character_status(character_id, now_time)
        if status_judge:
            cache.over_behavior_character[character_id] = 1


def character_target_judge(character_id: int, now_time: datetime.datetime):
    """
    查询角色可用目标活动并执行
    Keyword arguments:
    character_id -- 角色id
    """
    target, _, judge = search_target(character_id, list(game_config.config_target.keys()), set())
    if judge:
        target_config = game_config.config_target[target]
        cache.handle_state_machine_data[target_config.state_machine_id](character_id)
    else:
        start_time = cache.character_data[character_id].behavior.start_time
        now_judge = game_time.judge_date_big_or_small(start_time, now_time)
        if now_judge:
            cache.over_behavior_character[character_id] = 1
        else:
            next_time = game_time.get_sub_date(minute=1, old_date=start_time)
            cache.character_data[character_id].behavior.start_time = next_time


def judge_character_status(character_id: int, now_time: datetime.datetime) -> int:
    """
    校验并结算角色状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    bool -- 本次update时间切片内活动是否已完成
    """
    character_data = cache.character_data[character_id]
    start_time = character_data.behavior.start_time
    end_time = game_time.get_sub_date(minute=character_data.behavior.duration, old_date=start_time)
    time_judge = game_time.judge_date_big_or_small(now_time, end_time)
    add_time = (end_time.timestamp() - start_time.timestamp()) / 60
    character_data.status.setdefault(27, 0)
    character_data.status.setdefault(28, 0)
    character_data.status[27] += add_time * 0.02
    character_data.status[28] += add_time * 0.02
    if time_judge:
        talk.handle_talk(character_id)
        settle_behavior.handle_settle_behavior(character_id, now_time)
        character_data.behavior.behavior_id = constant.Behavior.SHARE_BLANKLY
        character_data.state = constant.CharacterStatus.STATUS_ARDER
    if time_judge == 1:
        character_data.behavior.start_time = end_time
        return 0
    elif time_judge == 2:
        character.init_character_behavior_start_time(character_id, now_time)
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
    now_test_judge = 0
    for target in target_list:
        if target in null_target:
            continue
        if target == 7:
            now_test_judge = 1
        target_premise_list = game_config.config_target_premise_data[target]
        now_weight = 0
        now_target_pass_judge = 0
        now_target_data = {}
        premise_judge = 1
        for premise in target_premise_list:
            premise_judge = handle_premise.handle_premise(premise, character_id)
            if premise_judge:
                now_weight += premise_judge
            else:
                premise_judge = 0
                if premise in game_config.config_effect_target_data:
                    now_target_list = game_config.config_effect_target_data[premise]
                    now_target, now_target_weight, now_judge = search_target(
                        character_id, now_target_list, null_target
                    )
                    if now_judge:
                        now_target_data.setdefault(now_target_weight, set())
                        now_target_data[now_target_weight].add(now_target)
                        now_weight += now_target_weight
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
            target_data.setdefault(now_weight, set())
            target_data[now_weight].add(target)
        else:
            now_max_weight = max(list(now_target_data))
            target_data.setdefault(now_max_weight, set())
            target_data[now_weight].add(random.choice(list(now_target_data[now_max_weight])))
    if len(target_data):
        max_weight = max(list(target_data.keys()))
        return random.choice(list(target_data[max_weight])), max_weight, 1
    return "", 0, 0
