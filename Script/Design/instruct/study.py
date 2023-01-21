import random
import datetime
from typing import List
from types import FunctionType
from Script.Core import cache_control, game_type, get_text
from Script.Design import update, character, course, game_time, constant, handle_instruct
from Script.Config import normal_config, game_config


cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
_: FunctionType = get_text._
""" 翻译api """
width: int = normal_config.config_normal.text_width
""" 屏幕宽度 """


@handle_instruct.add_instruct(
    constant.Instruct.VIEW_THE_SCHOOL_TIMETABLE,
    constant.InstructType.STUDY,
    _("查看课程表"),
    {},
)
def handle_view_school_timetable():
    """处理查看课程表指令"""
    cache.now_panel_id = constant.Panel.VIEW_SCHOOL_TIMETABLE


@handle_instruct.add_instruct(
    constant.Instruct.ATTEND_CLASS,
    constant.InstructType.STUDY,
    _("上课"),
    {
        constant.Premise.ATTEND_CLASS_TODAY,
        constant.Premise.IN_CLASSROOM,
        constant.Premise.IN_CLASS_TIME,
        constant.Premise.IS_STUDENT,
    },
)
def handle_attend_class():
    """处理上课指令"""
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data: game_type.Character = cache.character_data[0]
    end_time = 0
    school_id, phase = course.get_character_school_phase(0)
    now_time = datetime.datetime.fromtimestamp(cache.game_time, game_time.time_zone)
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
    if not now_course_index:
        now_course = random.choice(
            list(game_config.config_school_phase_course_data[school_id][phase])
        )
    else:
        now_course = cache.course_time_table_data[school_id][phase][now_week][now_course_index]
    character_data.behavior.duration = end_time
    character_data.behavior.behavior_id = constant.Behavior.ATTEND_CLASS
    character_data.state = constant.CharacterStatus.STATUS_ATTEND_CLASS
    character_data.behavior.course_id = now_course
    update.game_update_flow(end_time)


@handle_instruct.add_instruct(
    constant.Instruct.TEACH_A_LESSON,
    constant.InstructType.STUDY,
    _("教课"),
    {
        constant.Premise.ATTEND_CLASS_TODAY,
        constant.Premise.IN_CLASSROOM,
        constant.Premise.IN_CLASS_TIME,
        constant.Premise.IS_TEACHER,
    },
)
def handle_teach_a_lesson():
    """处理教课指令"""
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data: game_type.Character = cache.character_data[0]
    end_time = 0
    now_time = datetime.datetime.fromtimestamp(cache.game_time, game_time.time_zone)
    now_time_value = now_time.hour * 100 + now_time.minute
    now_week = now_time.weekday()
    timetable_list: List[game_type.TeacherTimeTable] = cache.teacher_school_timetable[0]
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
    update.game_update_flow(end_time)


@handle_instruct.add_instruct(
    constant.Instruct.SELF_STUDY,
    constant.InstructType.STUDY,
    _("自习"),
    {
        constant.Premise.IN_CLASSROOM,
        constant.Premise.IS_STUDENT,
    },
)
def handle_self_study():
    """处理自习指令"""
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data: game_type.Character = cache.character_data[0]
    school_id, phase = course.get_character_school_phase(0)
    now_course_list = list(game_config.config_school_phase_course_data[school_id][phase])
    now_course = random.choice(now_course_list)
    character_data.behavior.behavior_id = constant.Behavior.SELF_STUDY
    character_data.behavior.duration = 10
    character_data.state = constant.CharacterStatus.STATUS_SELF_STUDY
    character_data.behavior.course_id = now_course
    update.game_update_flow(10)
