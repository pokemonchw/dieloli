import datetime
from typing import List
from Script.Design import handle_premise, game_time, character, constant
from Script.Core import game_type, cache_control
from Script.Config import game_config

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.IN_BREAKFAST_TIME)
def handle_in_breakfast_time(character_id: int) -> int:
    """
    校验当前时间是否处于早餐时间段
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    now_time = game_time.get_sun_time(character_data.behavior.start_time)
    return now_time == 4


@handle_premise.add_premise(constant.Premise.IN_LUNCH_TIME)
def handle_in_lunch_time(character_id: int) -> int:
    """
    校验当前是否处于午餐时间段
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    now_time = game_time.get_sun_time(character_data.behavior.start_time)
    return now_time == 7


@handle_premise.add_premise(constant.Premise.IN_DINNER_TIME)
def handle_in_dinner_time(character_id: int) -> int:
    """
    校验当前是否处于晚餐时间段
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    now_time = game_time.get_sun_time(character_data.behavior.start_time)
    return now_time == 9


@handle_premise.add_premise(constant.Premise.IN_SLEEP_TIME)
def handle_in_sleep_time(character_id: int) -> int:
    """
    校验角色当前是否处于睡觉时间
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    now_time: datetime.datetime = datetime.datetime.fromtimestamp(
        character_data.behavior.start_time, game_time.time_zone
    )
    if now_time.hour >= 22 or now_time.hour <= 4:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IN_SIESTA_TIME)
def handle_in_siesta_time(character_id: int) -> int:
    """
    校验角色是否处于午休时间
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    now_time: datetime.datetime = datetime.datetime.fromtimestamp(
        character_data.behavior.start_time, game_time.time_zone
    )
    if now_time.hour >= 12 or now_time.hour <= 15:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.NO_IN_SLEEP_TIME)
def handle_no_in_sleep_time(character_id: int) -> int:
    """
    校验角色当前是否不处于睡觉时间
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    now_time: datetime.datetime = datetime.datetime.fromtimestamp(
        character_data.behavior.start_time, game_time.time_zone
    )
    if now_time.hour >= 22 or now_time.hour <= 4:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.ATTEND_CLASS_TODAY)
def handle_attend_class_today(character_id: int) -> int:
    """
    校验角色今日是否需要上课
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    return game_time.judge_attend_class_today(character_id)


@handle_premise.add_premise(constant.Premise.APPROACHING_CLASS_TIME)
def handle_approaching_class_time(character_id: int) -> int:
    """
    校验角色是否临近上课时间
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    now_time: datetime.datetime = datetime.datetime.fromtimestamp(
        character_data.behavior.start_time
    )
    now_time_value = now_time.hour * 100 + now_time.minute
    next_time = 0
    if character_data.age <= 18:
        school_id = 0
        if character_data.age in range(13, 16):
            school_id = 1
        elif character_data.age in range(16, 19):
            school_id = 2
        for session_id in game_config.config_school_session_data[school_id]:
            session_config = game_config.config_school_session[session_id]
            if (
                session_config.start_time > now_time_value
                and next_time == 0
                or session_config.start_time < next_time
            ):
                next_time = session_config.start_time
        if next_time == 0:
            return 0
    if character_id in cache.teacher_school_timetable:
        now_week = now_time.weekday()
        timetable_list: List[game_type.TeacherTimeTable] = cache.teacher_school_timetable[
            character_id
        ]
        for timetable in timetable_list:
            if timetable.week_day != now_week:
                continue
            if timetable.time > now_time_value and next_time == 0 or timetable.time < next_time:
                next_time = timetable.time
        if next_time == 0:
            return 0
    next_value = int(next_time / 100) * 60 + next_time % 100
    now_value = int(now_time_value / 100) * 60 + now_time_value % 100
    add_time = next_value - now_value
    if add_time > 30:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.IN_CLASS_TIME)
def handle_in_class_time(character_id: int) -> int:
    """
    校验角色是否处于上课时间
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    judge, _unused, _unused, _unused, _unused = character.judge_character_in_class_time(
        character_id
    )
    return judge


@handle_premise.add_premise(constant.Premise.TONIGHT_IS_FULL_MOON)
def handle_tonight_is_full_moon(character_id: int) -> int:
    """
    校验今夜是否是满月
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    now_time = character_data.behavior.start_time
    if not now_time:
        now_time = cache.game_time
    moon_phase = game_time.get_moon_phase(now_time)
    if moon_phase in {11, 12}:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_SPRING)
def handle_is_spring(character_id: int) -> int:
    """
    校验现在是否是春天
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    now_time = character_data.behavior.start_time
    if not now_time:
        now_time = cache.game_time
    solar_period = game_time.get_solar_period(now_time)
    season = game_config.config_solar_period[solar_period].season
    return not season


@handle_premise.add_premise(constant.Premise.IS_SUMMER)
def handle_is_summer(character_id: int) -> int:
    """
    校验现在是否是夏天
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    now_time = character_data.behavior.start_time
    if not now_time:
        now_time = cache.game_time
    solar_period = game_time.get_solar_period(now_time)
    season = game_config.config_solar_period[solar_period].season
    return season == 1
