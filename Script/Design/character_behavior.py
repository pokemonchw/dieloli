import time
import random
import concurrent.futures
from uuid import UUID
from concurrent.futures import ThreadPoolExecutor
from types import FunctionType
from typing import Dict, Set, List
import datetime
from Script.Core import cache_control, game_type, value_handle, get_text
from Script.Design import (
    character_handle, constant, game_time, settle_behavior, handle_premise, event,
    cooking, map_handle, attr_calculation,
)
from Script.Config import game_config, normal_config

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
handle_thread_pool: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=50)
""" 处理角色行为的线程池 """


def init_character_behavior():
    """
    角色行为树总控制
    """
    global time_index
    cache.character_target_data = {}
    cache.character_target_score_data = {}
    update_cafeteria()
    now_status_data = {}
    now_status_data[0] = set()
    now_status_data[1] = set()
    now_status_data[2] = set()
    for i in cache.character_data:
        if not i:
            continue
        now_judge = check_character_status_judge(i)
        now_status_data[now_judge].add(i)
        now_character_data = cache.character_data[i]
    for i in cache.character_data:
        cache.character_data[i].premise_data = {}
        if not i:
            continue
        status = check_character_status_judge(i)
        now_status_data[status].add(i)
    if len(now_status_data[1]) == len(cache.character_data) - 1:
        return
    now_list = [character_target_judge(i) for i in now_status_data[0]]
    for result in now_list:
        run_character_target(result)
    for i in now_status_data[2]:
        judge_character_status(i, cache.game_time)
    for i in cache.character_data:
        if not i:
            continue
        attr_calculation.update_character_mana_point_and_hp_point_max(i)
        judge_character_dead(i)
    refresh_teacher_classroom()
    time_index = 0
    player_data: game_type.Character = cache.character_data[0]
    if cache.game_time >= player_data.behavior.start_time + 60 * player_data.behavior.duration:
        judge_character_status(0, cache.game_time)
        attr_calculation.update_character_mana_point_and_hp_point_max(0)
        judge_character_dead(0)


def check_character_status_judge(character_id: int) -> int:
    """
    验证角色状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- (0:需要查找目标,1:跳过本轮计算,2:需要进行结算)
    """
    character_data: game_type.Character = cache.character_data[character_id]
    # 当角色死亡时跳过结算
    if character_data.state == constant.CharacterStatus.STATUS_DEAD:
        return 1
    # 当角色状态为闲置时需要查找目标
    if character_data.state == constant.CharacterStatus.STATUS_ARDER:
        if character_data.behavior.start_time == 0:
            character_data.behavior.start_time = cache.game_time
        return 0
    # 当行动所需时间为0时立刻进行结算
    if character_data.behavior.duration == 0:
        return 2
    start_time = character_data.behavior.start_time
    end_time = start_time + 60 * character_data.behavior.duration
    # 验证结算时间
    time_judge = game_time.judge_date_big_or_small(cache.game_time, end_time)
    if time_judge == 0:
        return 1
    return 2


def character_target_judge(character_id: int) -> game_type.ExecuteTarget:
    """
    查询角色可用目标活动并执行
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    game_type.Character -- 更新后的角色数据
    """
    global time_index
    now_time: int = cache.game_time
    target_weight_data = {}
    # 取出角色数据
    character_data: game_type.Character = cache.character_data[character_id]
    null_target_set = set()
    target, _, judge = search_target(
        character_id,
        set(game_config.config_target.keys()),
        null_target_set,
        target_weight_data,
        0,
        ""
    )
    if judge:
        target.imitate_character_id = character_id
        if target.affiliation == "":
            target.affiliation = target.uid
    if target == None:
        target = game_type.ExecuteTarget()
    target.character_id = character_id
    return target

def run_character_target(target: game_type.ExecuteTarget):
    """
    执行角色目标
    Keyword arguments:
    Return arguments:
    game_type.Character -- 更新后的角色数据
    """
    now_time: int = cache.game_time
    player_data: game_type.Character = cache.character_data[0]
    character_id: int = target.character_id
    character_data: game_type.Character = cache.character_data[target.character_id]
    if target.uid != "":
        if target.imitate_character_id != 0:
            if target.imitate_character_id in cache.character_target_data:
                cache.character_target_data[target.imitate_character_id].score += 1
                cache.character_target_score_data[target.imitate_character_id] += 1
        cache.character_target_data[character_id] = target
        cache.character_target_score_data[character_id] = 0
        target_config = game_config.config_target[target.uid]
        character_data.ai_target = target.affiliation
        constant.handle_state_machine_data[target_config.state_machine_id](character_id)
        event.handle_event(character_id, 1, now_time, now_time)
        cache.character_target_data[target.character_id] = target
    start_time = cache.character_data[character_id].behavior.start_time
    now_judge = game_time.judge_date_big_or_small(start_time, now_time)
    if now_judge:
        character_data.ai_target = 0


def judge_character_dead(character_id: int):
    """
    校验角色状态并处死角色
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        character_data.state = 13
        return
    if character_data.state == 13:
        character_data.dead = 1
        return
    character_data.status.setdefault(27, 0)
    character_data.status.setdefault(28, 0)
    dead_judge = 0
    while 1:
        if character_data.status[27] >= 100: # 饿死
            dead_judge = 1
            character_data.cause_of_death = 0
            break
        if character_data.status[28] >= 100: # 渴死
            dead_judge = 1
            character_data.cause_of_death = 1
            break
        if character_data.hit_point <= 0: # 累死
            dead_judge = 1
            character_data.cause_of_death = 2
            break
        if character_data.extreme_exhaustion_time: # 困死
            exhaustion_time = (cache.game_time - character_data.extreme_exhaustion_time) / 60
            sudden_death_probability = exhaustion_time * (100 / 8640)
            sudden_death_probability = max(sudden_death_probability, 100 / 8640)
            now_range = random.randint(0,100)
            if now_range < sudden_death_probability:
                dead_judge = 1
                character_data.cause_of_death = 3
        break
    if dead_judge:
        character_data.dead = 1
        character_data.state = 13


def judge_character_status(character_id: int, now_time: int) -> int:
    """
    校验并结算角色状态
    Keyword arguments:
    character_id -- 角色id
    now_time -- 指定时间戳
    Return arguments:
    bool -- 本次update时间切片内活动是否已完成
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.behavior.start_time == 0:
        character_data.behavior.start_time = cache.game_time
    start_time = character_data.behavior.start_time
    end_time = start_time + 60 * character_data.behavior.duration
    add_time = (end_time - start_time) / 60
    if not add_time:
        character_data.behavior = game_type.Behavior()
        character_data.behavior.start_time = now_time
        character_data.state = constant.CharacterStatus.STATUS_ARDER
        return 1
    # 增加饥饿和口渴
    last_hunger_time = start_time
    if character_data.last_hunger_time:
        last_hunger_time = character_data.last_hunger_time
    character_data.status.setdefault(27, 0)
    character_data.status.setdefault(28, 0)
    character_data.status[27] += 0.02 * (now_time - last_hunger_time) / 60
    character_data.status[28] += 0.02 * (now_time - last_hunger_time) / 60
    character_data.last_hunger_time = now_time
    # 增加疲惫
    if character_data.state != constant.CharacterStatus.STATUS_SLEEP:
        character_data.status.setdefault(25, 0)
        character_data.status[25] += 0.0694 * add_time
        if character_data.status[25] >= 100:
            if character_data.extreme_exhaustion_time == 0:
                character_data.extreme_exhaustion_time = cache.game_time
        else:
            if character_data.extreme_exhaustion_time != 0:
                character_data.extreme_exhaustion_time = 0
    character_data.behavior.temporary_status = game_type.TemporaryStatus()
    if cache.game_time >= end_time:
        event.handle_event(character_id, 0,cache.game_time, cache.game_time)
        character_data.behavior.start_time = cache.game_time
        character_data.behavior.duration = 0


def refresh_teacher_classroom():
    """ 刷新所有教师准备上课的班级位置 """
    for character_id in cache.teacher_class_time_data:
        character_data: game_type.Character = cache.character_data[character_id]
        identity_data: game_type.TeacherIdentity = character_data.identity_data[1]
        now_judge = game_time.judge_attend_class_today(character_id)
        if not now_judge:
            continue
        now_week = game_time.get_week_day(character_data.behavior.start_time)
        timetable_list: List[game_type.TeacherTimeTable] = cache.teacher_school_timetable[
            character_id
        ]
        time_judge = True
        now_time = datetime.datetime.fromtimestamp(character_data.behavior.start_time)
        now_time_value = now_time.hour * 100 + now_time.minute
        for timetable in timetable_list:
            if timetable.week_day != now_week:
                continue
            if now_time_value < timetable.time - 9:
                continue
            if now_time_value > timetable.end_time:
                continue
            time_judge = False
            identity_data.now_classroom = map_handle.get_map_system_path_str_for_list(timetable.class_room)
        if time_judge:
            identity_data.now_classroom = ""


def search_target(
        character_id: int,
        target_list: set,
        null_target: set,
        target_weight_data: Dict[int, int],
        sub_weight: int,
        original_target_id: str
) -> (game_type.ExecuteTarget, int, bool):
    """
    查找可用目标
    Keyword arguments:
    character_id -- 角色id
    target_list -- 检索的目标列表
    null_target -- 被排除的目标
    premise_data -- 已算出的前提权重
    target_weight_data -- 已算出权重的目标列表
    sub_weight -- 向下传递的目标权重
    original_target_id -- 原始的目标id
    Return arguments:
    game_type.ExecuteTarget -- 要执行的目标数据
    int -- 目标权重
    bool -- 前提是否能够被满足
    """
    character_data = cache.character_data[character_id]
    target_data: Dict[float, Set[game_type.ExecuteTarget]] = {}
    for target in target_list:
        if target in null_target:
            continue
        if target in target_weight_data:
            target_data.setdefault(target_weight_data[target], set())
            now_execute_target = game_type.ExecuteTarget()
            now_execute_target.uid = target
            now_execute_target.weight = target_weight_data[target] + sub_weight
            now_execute_target.affiliation = original_target_id
            target_data.setdefault(now_execute_target.weight, set())
            target_data[now_execute_target.weight].add(now_execute_target)
            continue
        target_config = game_config.config_target[target]
        if character_data.passive_sex:
            if target_config.needs_hierarchy > 1:
                continue
        if not len(target_config.premise):
            target_data.setdefault(1, set())
            now_execute_target = game_type.ExecuteTarget()
            now_execute_target.uid = target
            now_execute_target.weight = 1 + sub_weight + (500 - 100 * target_config.needs_hierarchy)
            now_execute_target.affiliation = original_target_id
            target_data[1].add(now_execute_target)
            target_weight_data[target] = now_execute_target.weight = 1
            continue
        now_weight = sub_weight + (500 - 100 * target_config.needs_hierarchy)
        now_target_pass_judge = 0
        now_target_data = {}
        premise_judge = 1
        null_premise_set = set()
        for premise in target_config.premise:
            if premise in character_data.premise_data:
                premise_judge = character_data.premise_data[premise]
            else:
                premise_judge = handle_premise.handle_premise(premise, character_id)
                premise_judge = max(premise_judge, 0)
                character_data.premise_data[premise] = premise_judge
            if premise_judge:
                now_weight += premise_judge
            else:
                if premise in game_config.config_effect_target_data and premise not in character_data.premise_data:
                    null_premise_set.add(premise)
                else:
                    now_target_pass_judge = 1
                    break
        if not now_target_pass_judge and null_premise_set:
            premise_judge = 0
            now_original_target_id = target
            if original_target_id != "":
                now_original_target_id = original_target_id
            for premise in null_premise_set:
                now_judge = 0
                now_target_weight = 0
                if premise in cache.character_premise_target_data:
                    if cache.character_premise_target_data[premise]:
                        conformity_judge = 0
                        if character_data.nature[1] > 50:
                            conformity = (character_data.nature[1] - 50) * 2
                            now_range = random.random() * 100
                            if now_range < conformity:
                                conformity_judge = 1
                        if conformity_judge:
                            now_target_id = value_handle.get_random_for_weight(
                                cache.character_premise_target_data[premise])
                            now_target, now_target_weight, conformity_judge = search_target(
                                character_id,
                                {now_target_id},
                                null_target,
                                target_weight_data,
                                now_original_target_id,
                            )
                if not now_judge:
                    now_target_list = game_config.config_effect_target_data[premise] - null_target
                    now_target_list.remove(target)
                    now_target, now_target_weight, now_judge = search_target(
                        character_id,
                        now_target_list,
                        null_target,
                        target_weight_data,
                        now_weight,
                        now_original_target_id,
                    )
                if now_judge:
                    now_target_data.setdefault(now_target_weight, set())
                    now_target_data[now_target_weight].add(now_target)
                    now_weight += now_target_weight
                    cache.character_premise_target_data.setdefault(premise, {})
                    cache.character_premise_target_data[premise].setdefault(now_target.uid, 0)
                    cache.character_premise_target_data[premise][now_target.uid] += 1
                else:
                    now_target_pass_judge = 1
                    break
        if now_target_pass_judge:
            null_target.add(target)
            target_weight_data[target] = 0
            continue
        if premise_judge:
            target_data.setdefault(now_weight, set())
            now_execute_target = game_type.ExecuteTarget()
            now_execute_target.uid = target
            now_execute_target.weight = now_weight
            now_execute_target.affiliation = original_target_id
            target_data[now_weight].add(now_execute_target)
            target_weight_data[target] = now_weight
        else:
            now_value_weight = value_handle.get_rand_value_for_value_region(now_target_data)
            target_data.setdefault(now_weight, set())
            target_data[now_weight].add(random.choice(now_target_data[now_value_weight]))
    if target_data:
        value_weight = value_handle.get_rand_value_for_value_region(target_data)
        return random.choice(list(target_data[value_weight])), value_weight, 1
    return None, 0, 0


def update_cafeteria():
    """刷新食堂内食物"""
    food_judge = 1
    game_time_object = datetime.datetime.fromtimestamp(cache.game_time)
    if game_time_object.hour not in {6, 7, 8, 12, 13, 18, 19}:
        if cache.restaurant_data != {}:
            cache.restaurant_data = {}
    else:
        for food_type in cache.restaurant_data:
            food_list: Dict[UUID, game_type.Food] = cache.restaurant_data[food_type]
            for food_id in food_list:
                food: game_type.Food = food_list[food_id]
                if food.eat:
                    food_judge = 0
                break
            if not food_judge:
                break
        if food_judge:
            cooking.init_restaurant_data()

