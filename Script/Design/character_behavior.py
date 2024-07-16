import time
import random
import concurrent.futures
from uuid import UUID
from concurrent.futures import ThreadPoolExecutor
from types import FunctionType
from typing import Dict, Set, List
import datetime
from Script.Core import cache_control, game_type, value_handle, get_text
from Script.Design import character_handle, constant, game_time, settle_behavior, handle_premise, event, cooking, map_handle
from Script.Config import game_config, normal_config
from Script.UI.Moudle import draw

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
_: FunctionType = get_text._
""" 翻译api """
window_width: int = normal_config.config_normal.text_width
""" 窗体宽度 """
handle_thread_pool: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=50)
""" 处理角色行为的线程池 """


def init_character_behavior():
    """
    角色行为树总控制
    """
    global time_index
    character_handle.build_similar_character_searcher()
    cache.character_target_data = {}
    cache.character_target_score_data = {}
    while 1:
        now_status_data = {}
        now_status_data[0] = set()
        now_status_data[1] = set()
        now_status_data[2] = set()
        for i in cache.character_data:
            if not i:
                continue
            now_judge = check_character_status_judge(i)
            now_status_data[now_judge].add(i)
        for i in cache.character_data:
            cache.character_data[i].premise_data = {}
            if not i:
                continue
            status = check_character_status_judge(i)
            now_status_data[status].add(i)
        if len(now_status_data[1]) == len(cache.character_data) - 1:
            break
        now_list = [character_target_judge(i) for i in now_status_data[0]]
        for result in now_list:
            run_character_target(result)
        for i in now_status_data[2]:
            judge_character_status(i, cache.game_time)
        for i in cache.character_data:
            if not i:
                continue
            judge_character_dead(i)
        refresh_teacher_classroom()
    time_index = 0
    judge_character_status(0, cache.game_time)
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
        if character_data.behavior.start_time < cache.game_time:
            return 0
        return 1
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
    # 计算角色向量
    character_vector = cache.character_vector_data[character_id - 1]
    # 找出最相似的角色
    near_character_list, distance_list = cache.similar_character_searcher.knn_query(character_vector, k=1)
    near_character_id = near_character_list[0][0]
    distance = distance_list[0][0]
    near_judge = 0
    if near_character_id in cache.character_target_data:
        near_target = cache.character_target_data[near_character_id]
        now_score = near_target.score * 0.0003
        now_range = random.random()
        if now_range <= 1 - distance + now_score:
            near_judge = 1
    # 获取最相似角色的行动目标
    target: game_type.ExecuteTarget = game_type.ExecuteTarget()
    judge = 0
    null_target_set = set()
    if near_judge:
        target, _, judge = search_target(
            character_id,
            {cache.character_target_data[near_character_id].affiliation},
            null_target_set,
            target_weight_data,
            0,
            ""
        )
        # 自身可以使用相似角色的目标时为相似角色进行加分
        if judge:
            target.imitate_character_id = near_character_id
    # 无法模仿相似角色时，若自身性格有合群倾向，则改为模仿最受欢迎的角色(即分值最高的角色)的行为
    if not judge:
        conformity_judge = 0
        if character_data.nature[1] > 50:
            if cache.character_target_data:
                conformity = (character_data.nature[1] - 50) * 2
                now_range = random.random() * 100
                if now_range < conformity:
                    conformity_judge = 1
        if conformity_judge and cache.character_target_score_data:
            imitate_character_id = value_handle.get_random_for_weight(cache.character_target_score_data)
            imitate_target_uid = cache.character_target_data[imitate_character_id].affiliation
            imitate_target_config = game_config.config_target[imitate_target_uid]
            target, _, judge = search_target(
                character_id,
                {cache.character_target_data[imitate_character_id].affiliation},
                null_target_set,
                target_weight_data,
                0,
                ""
            )
            if judge:
                target.imitate_character_id = imitate_character_id
                now_target_config = game_config.config_target[target.uid]
    if not judge:
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
    if judge:
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
        event_draw = event.handle_event(character_id, 1)
        if (not character_id) or (player_data.target_character_id == character_id):
            if event_draw is not None:
                event_draw.draw()
        cache.character_target_data[target.character_id] = target
    else:
        cache.character_data[character_id].behavior.start_time += 60
    start_time = cache.character_data[character_id].behavior.start_time
    now_judge = game_time.judge_date_big_or_small(start_time, now_time)
    if now_judge:
        cache.over_behavior_character.add(character_id)
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
        if character_id not in cache.over_behavior_character:
            cache.over_behavior_character.add(character_id)
        return
    if character_data.state == 13:
        character_data.dead = 1
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
        if character_data.extreme_exhaustion_time:
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
    if character_data.behavior.start_time == 0:
        character_data.behavior.start_time = cache.game_time
    start_time = character_data.behavior.start_time
    end_time = start_time + 60 * character_data.behavior.duration
    add_time = (end_time - start_time) / 60
    if not add_time:
        character_data.behavior = game_type.Behavior()
        character_data.behavior.start_time = end_time
        character_data.state = constant.CharacterStatus.STATUS_ARDER
        return 1
    # 增加饥饿和口渴
    if end_time < now_time:
        now_time = end_time
    last_hunger_time = start_time
    if character_data.last_hunger_time:
        last_hunger_time = character_data.last_hunger_time
    hunger_time = int((now_time - last_hunger_time) / 60)
    character_data.status.setdefault(27, 0)
    character_data.status.setdefault(28, 0)
    character_data.status[27] += hunger_time * 0.02
    character_data.status[28] += hunger_time * 0.02
    character_data.last_hunger_time = now_time
    # 增加疲惫
    if character_data.state != constant.CharacterStatus.STATUS_SLEEP:
        character_data.status.setdefault(25, 0)
        character_data.status[25] += add_time * 0.0694
        if character_data.status[25] >= 100:
            if character_data.extreme_exhaustion_time == 0:
                character_data.extreme_exhaustion_time = cache.game_time
        else:
            if character_data.extreme_exhaustion_time != 0:
                character_data.extreme_exhaustion_time = 0
    player_data: game_type.Character = cache.character_data[0]
    line_feed = draw.NormalDraw()
    line_feed.text = "\n"
    while 1:
        character_data.behavior.temporary_status = game_type.TemporaryStatus()
        event_draw = event.handle_event(character_id, 0)
        if event_draw == None:
            character_data.behavior.temporary_status = game_type.TemporaryStatus()
            character_data.behavior.behavior_id = constant.Behavior.SHARE_BLANKLY
            character_data.ai_target = 0
            character_data.behavior.move_target = []
            character_data.behavior.move_src = []
            character_data.behavior.start_time = now_time
            character_data.state = constant.CharacterStatus.STATUS_ARDER
            break
        settle_output = settle_behavior.handle_settle_behavior(character_id, end_time, event_draw.event_id)
        character_data.behavior.temporary_status = game_type.TemporaryStatus()
        character_data.behavior.behavior_id = constant.Behavior.SHARE_BLANKLY
        character_data.behavior.move_target = []
        character_data.behavior.move_src = []
        character_data.behavior.start_time = now_time
        character_data.state = constant.CharacterStatus.STATUS_ARDER
        character_data.ai_target = 0
        climax_draw = settlement_pleasant_sensation(character_id)
        if (not character_id) or (player_data.target_character_id == character_id):
            if event_draw.text != "":
                event_draw.draw()
            if settle_output is not None:
                if settle_output[1]:
                    name_draw = draw.NormalDraw()
                    name_draw.text = "\n" + character_data.name + ": "
                    name_draw.width = window_width
                    name_draw.draw()
                settle_output[0].draw()
                line_feed.draw()
            if climax_draw is not None:
                if not character_id or not character_data.target_character_id:
                    climax_draw.draw()
                    line_feed.draw()
        break


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
            if now_time > timetable.end_time:
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
        if not low_climax_text:
            low_climax_text += organ_config.name
        else:
            low_climax_text += "+" + organ_config.name
    if low_climax_text:
        low_climax_text += _("绝顶")
    hight_climax_text = ""
    for organ in climax_data[2]:
        organ_config = game_config.config_organ[organ]
        if not hight_climax_text:
            hight_climax_text += organ_config.name
        else:
            hight_climax_text += "+" + organ_config.name
    if hight_climax_text:
        hight_climax_text += _("强绝顶")
    draw_list = []
    if low_climax_text:
        draw_list.append(low_climax_text)
    if hight_climax_text:
        draw_list.append(hight_climax_text)
    if draw_list:
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

