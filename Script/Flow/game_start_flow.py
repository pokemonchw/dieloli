from Script.Design import (
    character_handle,
    map_handle,
    course,
    interest,
    cooking,
)
from Script.Core import cache_contorl


def init_game_start():
    """
    用于结束角色创建正式开始游戏的初始化流程
    """
    character_handle.init_character_dormitory()
    character_handle.init_character_position()
    course.init_phase_course_hour()
    interest.init_character_interest()
    course.init_character_knowledge()
    course.init_class_teacher()
    course.init_class_time_table()
    course.init_teacher_table()
    cooking.init_recipes()
    character_position = cache_contorl.character_data[0].position
    map_handle.character_move_scene(["0"], character_position, 0)
    cache_contorl.now_flow_id = "main"
