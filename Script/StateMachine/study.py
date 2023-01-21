import random
import datetime
from typing import List
from Script.Config import game_config
from Script.Design import handle_state_machine, course, game_time, constant
from Script.Core import cache_control, game_type

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_state_machine.add_state_machine(constant.StateMachine.ATTEND_CLASS)
def character_attend_class(character_id: int):
    """
    角色在教室上课
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.behavior.behavior_id = constant.Behavior.ATTEND_CLASS
    end_time = 0
    school_id, phase = course.get_character_school_phase(character_id)
    now_time = datetime.datetime.fromtimestamp(
        character_data.behavior.start_time, game_time.time_zone
    )
    now_time_value = now_time.hour * 100 + now_time.minute
    now_course_index = 0
    for session_id in game_config.config_school_session_data[school_id]:
        session_config = game_config.config_school_session[session_id]
        if session_config.start_time <= now_time_value <= session_config.end_time:
            now_value = int(now_time_value / 100) * 60 + now_time_value % 100
            end_value = int(session_config.end_time / 100) * 60 + session_config.end_time % 100
            end_time = end_value - now_value + 1
            now_course_index = session_config.session
            break
    now_week = now_time.weekday()
    if not now_course_index or now_course_index:
        now_course = random.choice(
            list(game_config.config_school_phase_course_data[school_id][phase])
        )
    else:
        now_course = cache.course_time_table_data[school_id][phase][now_week][now_course_index]
    character_data.behavior.duration = end_time
    character_data.behavior.behavior_id = constant.Behavior.ATTEND_CLASS
    character_data.state = constant.CharacterStatus.STATUS_ATTEND_CLASS
    character_data.behavior.course_id = now_course


@handle_state_machine.add_state_machine(constant.StateMachine.TEACH_A_LESSON)
def character_teach_lesson(character_id: int):
    """
    角色在教室教课
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.behavior.behavior_id = constant.Behavior.TEACHING
    end_time = 0
    now_time = datetime.datetime.fromtimestamp(
        character_data.behavior.start_time, game_time.time_zone
    )
    now_week = now_time.weekday()
    now_time_value = now_time.hour * 100 + now_time.minute
    timetable_list: List[game_type.TeacherTimeTable] = cache.teacher_school_timetable[character_id]
    course = 0
    end_time = 0
    for timetable in timetable_list:
        if timetable.week_day != now_week:
            continue
        if timetable.time <= now_time_value and timetable.end_time <= now_time_value:
            now_value = int(now_time_value / 100) * 60 + now_time_value % 100
            end_value = int(timetable.end_time / 100) * 60 + timetable.end_time % 100
            end_time = end_value - now_value + 1
            course = timetable.course
            break
    character_data.behavior.duration = end_time
    character_data.behavior.behavior_id = constant.Behavior.TEACHING
    character_data.state = constant.CharacterStatus.STATUS_TEACHING
    character_data.behavior.course_id = course


@handle_state_machine.add_state_machine(constant.StateMachine.SELF_STUDY)
def character_self_study(character_id: int):
    """
    角色在自习
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.behavior.behavior_id = constant.Behavior.SELF_STUDY
    school_id, phase = course.get_character_school_phase(character_id)
    now_course_list = list(game_config.config_school_phase_course_data[school_id][phase])
    now_course_id = random.choice(now_course_list)
    character_data.behavior.duration = 10
    character_data.behavior.course_id = now_course_id
    character_data.state = constant.CharacterStatus.STATUS_SELF_STUDY


@handle_state_machine.add_state_machine(constant.StateMachine.PLAY_PIANO)
def character_play_piano(character_id: int):
    """
    角色弹奏钢琴
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.behavior.behavior_id = constant.Behavior.PLAY_PIANO
    character_data.behavior.duration = 30
    character_data.state = constant.CharacterStatus.STATUS_PLAY_PIANO


@handle_state_machine.add_state_machine(constant.StateMachine.SINGING)
def character_singing(character_id: int):
    """
    唱歌
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.behavior.behavior_id = constant.Behavior.SINGING
    character_data.behavior.duration = 5
    character_data.state = constant.CharacterStatus.STATUS_SINGING


@handle_state_machine.add_state_machine(constant.StateMachine.PLAY_GUITAR)
def character_play_guitar(character_id: int):
    """
    弹吉他
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.behavior.behavior_id = constant.Behavior.PLAY_GUITAR
    character_data.behavior.duration = 10
    character_data.state = constant.CharacterStatus.STATUS_PLAY_GUITAR


@handle_state_machine.add_state_machine(constant.StateMachine.DANCE)
def character_dance(character_id: int):
    """
    跳舞
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.behavior.behavior_id = constant.Behavior.DANCE
    character_data.behavior.duration = 10
    character_data.state = constant.CharacterStatus.STATUS_DANCE
