import random
from uuid import UUID
from types import FunctionType
from typing import Dict, Set
from Script.Core import (
    cache_control,
    game_path_config,
    game_type,
    constant,
    value_handle,
    get_text,
)
from Script.Design import (
    settle_behavior,
    game_time,
    handle_premise,
    event,
    cooking,
)
from Script.Config import game_config, normal_config
from Script.UI.Moudle import draw

game_path = game_path_config.game_path
cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
_: FunctionType = get_text._
""" 翻译api """
window_width: int = normal_config.config_normal.text_width
""" 窗体宽度 """


def init_character_behavior():
    """
    角色行为树总控制
    """
    while 1:
        if len(cache.over_behavior_character) >= len(cache.character_data):
            break
        for character_id in cache.character_data:
            if character_id in cache.over_behavior_character:
                continue
            character_behavior(character_id, cache.game_time)
            judge_character_dead(character_id)
        update_cafeteria()
    cache.over_behavior_character = set()


def update_cafeteria():
    """刷新食堂内食物"""
    food_judge = 1
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


def character_behavior(character_id: int, now_time: int):
    """
    角色行为控制
    Keyword arguments:
    character_id -- 角色id
    now_time -- 指定时间戳
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    if not character_data.behavior.start_time:
        character_data.behavior.start_time = now_time
    if character_data.state == constant.CharacterStatus.STATUS_ARDER:
        if character_id:
            character_target_judge(character_id, now_time)
        else:
            cache.over_behavior_character.add(0)
    else:
        status_judge = judge_character_status(character_id, now_time)
        if status_judge:
            cache.over_behavior_character.add(character_id)


def character_target_judge(character_id: int, now_time: int):
    """
    查询角色可用目标活动并执行
    Keyword arguments:
    character_id -- 角色id
    now_time -- 指定时间戳
    """
    premise_data = {}
    target_weight_data = {}
    target, _, judge = search_target(
        character_id,
        list(game_config.config_target.keys()),
        set(),
        premise_data,
        target_weight_data,
    )
    if judge:
        target_config = game_config.config_target[target]
        constant.handle_state_machine_data[target_config.state_machine_id](character_id)
        event_draw = event.handle_event(character_id, 1)
        if event_draw is not None:
            event_draw.draw()
    else:
        start_time = cache.character_data[character_id].behavior.start_time
        now_judge = game_time.judge_date_big_or_small(start_time, now_time)
        if now_judge:
            cache.over_behavior_character.add(character_id)
        else:
            cache.character_data[character_id].behavior.start_time += 60


def judge_character_dead(character_id: int):
    """
    校验角色状态并处死角色
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        if character_id not in cache.over_behavior_character:
            cache.over_behavior_character.add(character_id)
        return
    character_data.status.setdefault(27, 0)
    character_data.status.setdefault(28, 0)
    dead_judge = 0
    while 1:
        if character_data.status[27] >= 100:
            dead_judge = 1
            character_data.cause_of_death = 0
            break
        if character_data.status[27] >= 100:
            dead_judge = 1
            character_data.cause_of_death = 1
            break
        if character_data.hit_point <= 0:
            dead_judge = 1
            character_data.cause_of_death = 2
            break
        break
    if dead_judge:
        character_data.dead = 1
        character_data.state = 13
        if character_id not in cache.over_behavior_character:
            cache.over_behavior_character.add(character_id)


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
    start_time = character_data.behavior.start_time
    end_time = start_time + 60 * character_data.behavior.duration
    time_judge = game_time.judge_date_big_or_small(now_time, end_time)
    add_time = (end_time - start_time) / 60
    if not add_time:
        character_data.behavior = game_type.Behavior()
        character_data.behavior.start_time = end_time
        character_data.state = constant.CharacterStatus.STATUS_ARDER
        return 1
    last_hunger_time = start_time
    if character_data.last_hunger_time:
        last_hunger_time = character_data.last_hunger_time
    hunger_time = int((now_time - last_hunger_time) / 60)
    character_data.status.setdefault(27, 0)
    character_data.status.setdefault(28, 0)
    character_data.status[27] += hunger_time * 0.02
    character_data.status[28] += hunger_time * 0.02
    character_data.last_hunger_time = now_time
    line_feed = draw.NormalDraw()
    line_feed.text = "\n"
    if time_judge:
        character_data.behavior.temporary_status = game_type.TemporaryStatus()
        event_draw = event.handle_event(character_id, 0)
        if event_draw is not None:
            event_draw.draw()
        settle_output = settle_behavior.handle_settle_behavior(character_id, end_time, event_draw.event_id)
        if settle_output is not None:
            if settle_output[1]:
                name_draw = draw.NormalDraw()
                name_draw.text = "\n" + character_data.name + ": "
                name_draw.width = window_width
                name_draw.draw()
            settle_output[0].draw()
            line_feed.draw()
        character_data.behavior.temporary_status = game_type.TemporaryStatus()
        climax_draw = settlement_pleasant_sensation(character_id)
        if climax_draw is not None:
            if not character_id or not character_data.target_character_id:
                climax_draw.draw()
                line_feed.draw()
                event_draw = event.handle_event(character_id, 0)
                if event_draw is not None:
                    event_draw.draw()
        character_data.behavior.temporary_status = game_type.TemporaryStatus()
        character_data.behavior = game_type.Behavior()
        character_data.state = constant.CharacterStatus.STATUS_ARDER
    if time_judge == 1:
        character_data.behavior.start_time = end_time
        return 0
    if time_judge == 2:
        character_data.behavior.start_time = now_time
        return 0
    return 1


def search_target(
    character_id: int,
    target_list: list,
    null_target: set,
    premise_data: Dict[int, int],
    target_weight_data: Dict[int, int],
) -> (int, int, bool):
    """
    查找可用目标
    Keyword arguments:
    character_id -- 角色id
    target_list -- 检索的目标列表
    null_target -- 被排除的目标
    premise_data -- 已算出的前提权重
    target_weight_data -- 已算出权重的目标列表
    Return arguments:
    int -- 目标id
    int -- 目标权重
    bool -- 前提是否能够被满足
    """
    target_data = {}
    for target in target_list:
        if target in null_target:
            continue
        if target in target_weight_data:
            target_data.setdefault(target_weight_data[target], set())
            target_data[target_weight_data[target]].add(target)
            continue
        target_config = game_config.config_target[target]
        if not len(target_config.premise):
            target_data.setdefault(1, set())
            target_data[1].add(target)
            target_weight_data[target] = 1
            continue
        now_weight = 0
        now_target_pass_judge = 0
        now_target_data = {}
        premise_judge = 1
        for premise in target_config.premise:
            premise_judge = 0
            if premise in premise_data:
                premise_judge = premise_data[premise]
            else:
                premise_judge = handle_premise.handle_premise(premise, character_id)
                premise_judge = max(premise_judge, 0)
                premise_data[premise] = premise_judge
            if premise_judge:
                now_weight += premise_judge
            else:
                if premise in game_config.config_effect_target_data and premise not in premise_data:
                    now_target_list = game_config.config_effect_target_data[premise] - null_target
                    now_target, now_target_weight, now_judge = search_target(
                        character_id,
                        now_target_list,
                        null_target,
                        premise_data,
                        target_weight_data,
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
            target_weight_data[target] = 0
            continue
        if premise_judge:
            target_data.setdefault(now_weight, set())
            target_data[now_weight].add(target)
            target_weight_data[target] = now_weight
        else:
            now_value_weight = value_handle.get_rand_value_for_value_region(now_target_data.keys())
            target_data.setdefault(now_weight, set())
            target_data[now_weight].add(random.choice(list(now_target_data[now_value_weight])))
    if target_data:
        value_weight = value_handle.get_rand_value_for_value_region(target_data.keys())
        return random.choice(list(target_data[value_weight])), value_weight, 1
    return "", 0, 0


def settlement_pleasant_sensation(character_id: int) -> draw.NormalDraw():
    """
    结算角色快感
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    draw.NormalDraw -- 高潮结算绘制文本
    """
    character_data: game_type.Character = cache.character_data[character_id]
    behavior_data: game_type.Behavior = character_data.behavior
    climax_data: Dict[int, Set] = {1: set(), 2: set()}
    for organ in character_data.sex_experience:
        organ_config = game_config.config_organ[organ]
        status_id = organ_config.status_id
        if status_id in character_data.status:
            if character_data.status[status_id] >= 1000:
                now_level = 1
                if character_data.status[status_id] >= 10000:
                    now_level = 2
                if status_id == 0:
                    behavior_data.temporary_status.mouth_climax = now_level
                elif status_id == 1:
                    behavior_data.temporary_status.chest_climax = now_level
                elif status_id == 2:
                    behavior_data.temporary_status.vagina_climax = now_level
                elif status_id == 3:
                    behavior_data.temporary_status.clitoris_climax = now_level
                elif status_id == 4:
                    behavior_data.temporary_status.anus_climax = now_level
                elif status_id == 5:
                    behavior_data.temporary_status.penis_climax = now_level
                if now_level == 1:
                    character_data.status[21] *= 0.5
                elif now_level == 2:
                    character_data.status[21] *= 0.8
                character_data.status[21] = max(character_data.status[21], 0)
                character_data.status[status_id] = 0
                climax_data[now_level].add(organ)
                if now_level == 1:
                    character_data.sex_experience[organ] *= 1.01
                else:
                    character_data.sex_experience[organ] *= 1.1
    low_climax_text = ""
    for organ in climax_data[1]:
        organ_config = game_config.config_organ[organ]
        if not len(low_climax_text):
            low_climax_text += organ_config.name
        else:
            low_climax_text += "+" + organ_config.name
    if len(low_climax_text):
        low_climax_text += _("绝顶")
    hight_climax_text = ""
    for organ in climax_data[2]:
        organ_config = game_config.config_organ[organ]
        if not len(hight_climax_text):
            hight_climax_text += organ_config.name
        else:
            hight_climax_text += "+" + organ_config.name
    if len(hight_climax_text):
        hight_climax_text += _("强绝顶")
    draw_list = []
    if len(low_climax_text):
        draw_list.append(low_climax_text)
    if len(hight_climax_text):
        draw_list.append(hight_climax_text)
    if len(draw_list):
        draw_list.insert(0, character_data.name + ":")
        draw_text = ""
        for i in range(len(draw_list)):
            if i:
                draw_text += " " + draw_list[i]
            else:
                draw_text += draw_list[i]
        now_draw = draw.NormalDraw()
        now_draw.text = draw_text
        now_draw.width = window_width
        return now_draw
    return None
