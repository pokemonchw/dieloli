import datetime
from typing import List
from uuid import UUID
from Script.Design import handle_premise, map_handle, game_time, course, constant
from Script.Core import game_type, cache_control
from Script.Config import game_config

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.IN_SQUARE)
def handle_in_square(character_id: int) -> int:
    """
    校验角色是否处于操场中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    now_position = character_data.position
    now_scene_str = map_handle.get_map_system_path_str_for_list(now_position)
    now_scene_data = cache.scene_data[now_scene_str]
    if now_scene_data.scene_tag == "Square":
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IN_CAFETERIA)
def handle_in_cafeteria(character_id: int) -> int:
    """
    校验角色是否处于取餐区中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    now_position = character_data.position
    now_scene_str = map_handle.get_map_system_path_str_for_list(now_position)
    now_scene_data = cache.scene_data[now_scene_str]
    if now_scene_data.scene_tag == "Cafeteria":
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IN_RESTAURANT)
def handle_in_restaurant(character_id: int) -> int:
    """
    校验角色是否处于就餐区中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    now_position = character_data.position
    now_scene_str = map_handle.get_map_system_path_str_for_list(now_position)
    now_scene_data = cache.scene_data[now_scene_str]
    if now_scene_data.scene_tag == "Restaurant":
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IN_SWIMMING_POOL)
def handle_in_swimming_pool(character_id: int) -> int:
    """
    校验角色是否在游泳池中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    now_position = character_data.position
    now_scene_str = map_handle.get_map_system_path_str_for_list(now_position)
    now_scene_data = cache.scene_data[now_scene_str]
    if now_scene_data.scene_tag == "SwimmingPool":
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IN_CLASSROOM)
def handle_in_classroom(character_id: int) -> int:
    """
    校验角色是否处于所属教室中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    now_position = character_data.position
    now_scene_str = map_handle.get_map_system_path_str_for_list(now_position)
    if 0 in character_data.identity_data:
        identity_data: game_type.StudentIdentity = character_data.identity_data[0]
        if now_scene_str == identity_data.classroom:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.IN_SHOP)
def handle_in_shop(character_id: int) -> int:
    """
    校验角色是否在商店中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    now_position = character_data.position
    now_scene_str = map_handle.get_map_system_path_str_for_list(now_position)
    now_scene_data = cache.scene_data[now_scene_str]
    if now_scene_data.scene_tag == "Shop":
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IN_TOILET)
def handle_in_toilet(character_id: int) -> int:
    """
    校验角色是否在洗手间中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    now_position = character_data.position
    now_scene_str = map_handle.get_map_system_path_str_for_list(now_position)
    now_scene_data = cache.scene_data[now_scene_str]
    if now_scene_data.scene_tag == "Toilet":
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.NOT_IN_TOILET)
def handle_not_in_toilet(character_id: int) -> int:
    """
    校验角色是否不在洗手间中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    now_position = character_data.position
    now_scene_str = map_handle.get_map_system_path_str_for_list(now_position)
    now_scene_data = cache.scene_data[now_scene_str]
    if now_scene_data.scene_tag == "Toilet":
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.IN_PLAYER_SCENE)
def handle_in_player_scene(character_id: int) -> int:
    """
    校验角色是否与玩家处于同场景中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    now_character_data: game_type.Character = cache.character_data[character_id]
    if now_character_data.position == cache.character_data[0].position:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.SCENE_HAVE_OTHER_CHARACTER)
def handle_scene_have_other_target(character_id: int) -> int:
    """
    校验场景里是否有其他角色
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    scene_path = map_handle.get_map_system_path_str_for_list(character_data.position)
    scene_data: game_type.Scene = cache.scene_data[scene_path]
    return len(scene_data.character_list) > 1


@handle_premise.add_premise(constant.Premise.IN_DORMITORY)
def handle_in_dormitory(character_id: int) -> int:
    """
    校验角色是否在宿舍中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    now_position = map_handle.get_map_system_path_str_for_list(character_data.position)
    return now_position == character_data.dormitory


@handle_premise.add_premise(constant.Premise.NOT_IN_DORMITORY)
def handle_not_in_dormitory(character_id: int) -> int:
    """
    校验角色是否不在宿舍中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    now_position = map_handle.get_map_system_path_str_for_list(character_data.position)
    return not now_position == character_data.dormitory


@handle_premise.add_premise(constant.Premise.IN_TARGET_DORMITORY)
def handle_in_target_dormitory(character_id: int) -> int:
    """
    校验角色是否处于交互对象的宿舍中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    now_position = map_handle.get_map_system_path_str_for_list(character_data.position)
    return now_position == target_data.dormitory


@handle_premise.add_premise(constant.Premise.IN_MUSIC_CLASSROOM)
def handle_in_music_classroom(character_id: int) -> int:
    """
    校验角色是否处于音乐活动室中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    now_position = character_data.position
    now_scene_str = map_handle.get_map_system_path_str_for_list(now_position)
    now_scene_data = cache.scene_data[now_scene_str]
    if now_scene_data.scene_tag == "MusicClassroom":
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.NOT_IN_MUSIC_CLASSROOM)
def handle_not_in_music_classroom(character_id: int) -> int:
    """
    校验角色是否未处于音乐活动室中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    now_position = character_data.position
    now_scene_str = map_handle.get_map_system_path_str_for_list(now_position)
    now_scene_data = cache.scene_data[now_scene_str]
    if now_scene_data.scene_tag == "MusicClassroom":
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.SCENE_NO_HAVE_OTHER_CHARACTER)
def handle_scene_no_have_other_character(character_id: int) -> int:
    """
    校验场景中没有自己外的其他角色
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    scene_path = map_handle.get_map_system_path_str_for_list(character_data.position)
    scene_data: game_type.Scene = cache.scene_data[scene_path]
    return len(scene_data.character_list) == 1


@handle_premise.add_premise(constant.Premise.SCENE_CHARACTER_ONLY_PLAYER_AND_ONE)
def handle_scene_character_only_player_and_one(character_id: int) -> int:
    """
    校验场景中是否只有包括玩家在内的两个角色
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    now_position = character_data.position
    now_scene_str = map_handle.get_map_system_path_str_for_list(now_position)
    now_scene_data: game_type.Scene = cache.scene_data[now_scene_str]
    if 0 not in now_scene_data.character_list:
        return 0
    return len(now_scene_data.character_list) == 2


@handle_premise.add_premise(constant.Premise.BEYOND_FRIENDSHIP_TARGET_IN_SCENE)
def handle_beyond_friendship_target_in_scene(character_id: int) -> int:
    """
    校验是否对场景中某个角色抱有超越友谊的想法
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    now_position = character_data.position
    now_scene_str = map_handle.get_map_system_path_str_for_list(now_position)
    now_scene_data: game_type.Scene = cache.scene_data[now_scene_str]
    for i in {3, 4, 5}:
        character_data.social_contact.setdefault(i, set())
        for c in character_data.social_contact[i]:
            if c in now_scene_data.character_list:
                return 1
    return 0


@handle_premise.add_premise(constant.Premise.IN_FOUNTAIN)
def handle_in_fountain(character_id: int) -> int:
    """
    校验角色是否在喷泉场景中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    return character_data.position == ["8"]


@handle_premise.add_premise(constant.Premise.IN_LIBRARY)
def handle_in_library(character_id: int) -> int:
    """
    校验角色是否处于图书馆中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    return character_data.position[0] == "13"


@handle_premise.add_premise(constant.Premise.HAVE_OTHER_TARGET_IN_SCENE)
def handle_have_other_target_in_scene(character_id: int) -> int:
    """
    校验场景中是否有自己和交互对象以外的其他人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    scene_path_str = map_handle.get_map_system_path_str_for_list(character_data.position)
    scene_data: game_type.Scene = cache.scene_data[scene_path_str]
    return len(scene_data.character_list) > 2


@handle_premise.add_premise(constant.Premise.NO_HAVE_OTHER_TARGET_IN_SCENE)
def handle_no_have_other_target_in_scene(character_id: int) -> int:
    """
    校验场景中是否没有自己和交互对象以外的其他人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    scene_path_str = map_handle.get_map_system_path_str_for_list(character_data.position)
    scene_data: game_type.Scene = cache.scene_data[scene_path_str]
    return len(scene_data.character_list) <= 2


@handle_premise.add_premise(constant.Premise.HAVE_LIKE_TARGET_IN_SCENE)
def handle_have_like_target_in_scene(character_id: int) -> int:
    """
    校验是否有喜欢的人在场景中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(4, set())
    character_data.social_contact.setdefault(5, set())
    if not character_data.social_contact[4] and not character_data.social_contact[5]:
        return 0
    scene_path_str = map_handle.get_map_system_path_str_for_list(character_data.position)
    scene_data: game_type.Scene = cache.scene_data[scene_path_str]
    character_list = []
    for i in {4, 5}:
        for c in character_data.social_contact[i]:
            if c in scene_data.character_list:
                character_list.append(c)
    return len(character_list) > 0


@handle_premise.add_premise(constant.Premise.HAVE_NO_FIRST_KISS_LIKE_TARGET_IN_SCENE)
def handle_have_no_first_kiss_like_target_in_scene(character_id: int):
    """
    校验是否有自己喜欢的初吻还在的人在场景中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(4, set())
    character_data.social_contact.setdefault(5, set())
    scene_path_str = map_handle.get_map_system_path_str_for_list(character_data.position)
    scene_data: game_type.Scene = cache.scene_data[scene_path_str]
    character_list = []
    for i in {4, 5}:
        for c in character_data.social_contact[i]:
            if c in scene_data.character_list:
                c_data: game_type.Character = cache.character_data[c]
                if c_data.first_kiss == -1:
                    character_list.append(c)
    return len(character_list) > 0


@handle_premise.add_premise(constant.Premise.HAVE_DISLIKE_TARGET_IN_SCENE)
def handle_have_dislike_target_in_scene(character_id: int) -> int:
    """
    校验是否有讨厌的人在场景中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    scene_path_str = map_handle.get_map_system_path_str_for_list(character_data.position)
    scene_data: game_type.Scene = cache.scene_data[scene_path_str]
    character_list = []
    for i in {0, 1, 2, 3, 4}:
        character_data.social_contact.setdefault(i, set())
        for c in character_data.social_contact[i]:
            if c in scene_data.character_list:
                return 1
    return 0


@handle_premise.add_premise(constant.Premise.NOT_HAS_DISLIKE_TARGET_IN_SCENE)
def handle_not_has_dislike_target_in_scene(character_id: int) -> int:
    """
    校验场景中是否没有自己讨厌的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    scene_path_str = map_handle.get_map_system_path_str_for_list(character_data.position)
    scene_data: game_type.Scene = cache.scene_data[scene_path_str]
    for i in {0, 1, 2, 3, 4}:
        character_data.social_contact.setdefault(i, set())
        for c in character_data.social_contact[i]:
            if c in scene_data.character_list:
                return 0
    return 1


@handle_premise.add_premise(constant.Premise.NO_IN_CLASSROOM)
def handle_no_in_classroom(character_id: int) -> int:
    """
    校验角色是否不在所属教室中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    now_position = character_data.position
    now_scene_str = map_handle.get_map_system_path_str_for_list(now_position)
    if 0 in character_data.identity_data:
        identity_data: game_type.StudentIdentity = character_data.identity_data[0]
        if now_scene_str == identity_data.classroom:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TEACHER_NO_IN_CLASSROOM)
def handle_teacher_no_in_classroom(character_id: int) -> int:
    """
    校验角色所属班级的老师是否不在教室中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    if not game_time.judge_attend_class_today(character_id):
        return 0
    character_data: game_type.Character = cache.character_data[character_id]
    if 0 not in character_data.identity_data:
        return 0
    identity_data: game_type.StudentIdentity = character_data.identity_data[0]
    classroom: game_type.Scene = cache.scene_data[identity_data.classroom]
    now_time: datetime.datetime = datetime.datetime.fromtimestamp(
        character_data.behavior.start_time, game_time.time_zone
    )
    now_week = now_time.weekday()
    school_id, phase = course.get_character_school_phase(character_id)
    now_time_value = now_time.hour * 100 + now_time.minute
    if now_week in cache.course_time_table_data[school_id][phase]:
        now_course_index = 0
        next_time = 0
        for session_config_id in game_config.config_school_session_data[school_id]:
            session_config = game_config.config_school_session[session_config_id]
            if not next_time:
                if session_config.start_time >= now_time_value:
                    next_time = session_config.start_time
                    now_course_index = session_config.session
                elif session_config.end_time >= now_time_value:
                    next_time = session_config.end_time
                    now_course_index = session_config.session
                continue
            if (
                    session_config.start_time >= now_time_value
                    and session_config.start_time < next_time
            ):
                next_time = session_config.start_time
                now_course_index = session_config.session
            elif session_config.end_time >= now_time_value and session_config.end_time < next_time:
                next_time = session_config.start_time
                now_course_index = session_config.session
        if school_id not in cache.class_timetable_teacher_data:
            return 1
        if phase not in cache.class_timetable_teacher_data[school_id]:
            return 1
        if identity_data.classroom not in cache.class_timetable_teacher_data[school_id][phase]:
            return 1
        if (
                now_week
                not in cache.class_timetable_teacher_data[school_id][phase][identity_data.classroom]
        ):
            return 1
        if (
                now_course_index
                not in cache.class_timetable_teacher_data[school_id][phase][identity_data.classroom][
            now_week
        ]
        ):
            return 1
        now_teacher = cache.class_timetable_teacher_data[school_id][phase][
            identity_data.classroom
        ][now_week][now_course_index]
        if now_teacher not in classroom.character_list:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TEACHER_IN_CLASSROOM)
def handle_teacher_in_classroom(character_id: int) -> int:
    """
    校验角色所属班级的老师是否在教室中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    return not handle_premise.handle_premise(constant.Premise.TEACHER_NO_IN_CLASSROOM, character_id)


@handle_premise.add_premise(constant.Premise.IS_BEYOND_FRIENDSHIP_TARGET_IN_SCENE)
def handle_is_beyond_friendship_target_in_scene(character_id: int) -> int:
    """
    校验场景中是否有角色对自己抱有超越友谊的想法
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    scene_path_str = map_handle.get_map_system_path_str_for_list(character_data.position)
    scene_data: game_type.Scene = cache.scene_data[scene_path_str]
    now_weight = 0
    for now_character in scene_data.character_list:
        if now_character == character_id:
            continue
        now_character_data: game_type.Character = cache.character_data[now_character]
        if (
                character_id in now_character_data.social_contact_data
                and now_character_data.social_contact_data[character_id] > 8
        ):
            now_weight += now_character_data.social_contact_data[character_id]
    return now_weight > 0


@handle_premise.add_premise(constant.Premise.HAVE_STUDENTS_IN_CLASSROOM)
def handle_have_students_in_classroom(character_id: int) -> int:
    """
    校验是否有所教班级的学生在教室中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.age <= 18:
        return 0
    if character_id not in cache.teacher_school_timetable:
        return 0
    now_date: datetime.datetime = datetime.datetime.fromtimestamp(
        character_data.behavior.start_time
    )
    now_week = now_date.weekday()
    now_classroom = []
    now_time = 0
    timetable_list: List[game_type.TeacherTimeTable] = cache.teacher_school_timetable[character_id]
    now_time_value = now_date.hour * 100 + now_date.minute
    for timetable in timetable_list:
        if timetable.week_day != now_week:
            continue
        if now_time == 0:
            if timetable.time >= now_time_value:
                now_time = timetable.time
                now_classroom = timetable.class_room
            elif timetable.end_time >= now_time_value:
                now_time = timetable.end_time
                now_classroom = timetable.class_room
                break
            continue
        if timetable.time >= now_time_value and timetable.time < now_time:
            now_time = timetable.time
            now_classroom = timetable.class_room
            continue
        if timetable.end_time >= now_time_value and timetable.end_time < now_time:
            now_time = timetable.end_time
            now_classroom = timetable.class_room
    now_room_path_str = map_handle.get_map_system_path_str_for_list(now_classroom)
    now_scene_data: game_type.Scene = cache.scene_data[now_room_path_str]
    class_data = cache.classroom_students_data[now_room_path_str]
    return len(class_data & now_scene_data.character_list) > 0


@handle_premise.add_premise(constant.Premise.IN_ROOFTOP_SCENE)
def handle_in_rooftop_scene(character_id: int) -> int:
    """
    校验是否处于天台场景
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    now_position = character_data.position
    now_scene_str = map_handle.get_map_system_path_str_for_list(now_position)
    now_scene_data = cache.scene_data[now_scene_str]
    if now_scene_data.scene_tag == "Rooftop":
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IN_GROVE)
def handle_in_grove(character_id: int) -> int:
    """
    校验角色是否处于小树林中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    now_position = character_data.position
    if now_position[0] == "7":
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.NO_IN_GROVE)
def handle_no_in_grove(character_id: int) -> int:
    """
    校验角色是否未处于小树林中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    now_position = character_data.position
    if now_position[0] != "7":
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.NAKED_CHARACTER_IN_SCENE)
def handle_naked_character_in_scene(character_id: int) -> int:
    """
    校验场景中是否有人一丝不挂
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    scene_path = map_handle.get_map_system_path_str_for_list(character_data.position)
    scene_data: game_type.Scene = cache.scene_data[scene_path]
    for now_character in scene_data.character_list:
        now_character_data: game_type.Character = cache.character_data[now_character]
        for i in now_character_data.put_on:
            if isinstance(character_data.put_on[i], UUID):
                return 0
    return 1


@handle_premise.add_premise(constant.Premise.NO_IN_ITEM_SHOP)
def handle_no_in_item_shop(character_id: int) -> int:
    """
    校验角色是否不在超市中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    return character_data.position != ["11"]


@handle_premise.add_premise(constant.Premise.IN_ITEM_SHOP)
def handle_in_item_shop(character_id: int) -> int:
    """
    校验角色是否在超市中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    return character_data.position == ["11"]


@handle_premise.add_premise(constant.Premise.IN_STUDENT_UNION_OFFICE)
def handle_in_student_union_office(character_id: int) -> int:
    """
    校验角色是否在学生会办公室中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    return character_data.position == ["3", "1", "4"]


@handle_premise.add_premise(constant.Premise.NO_IN_CAFETERIA)
def handle_no_in_cafeteria(character_id: int) -> int:
    """
    校验角色是否未处于取餐区中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    now_position = character_data.position
    now_scene_str = map_handle.get_map_system_path_str_for_list(now_position)
    now_scene_data = cache.scene_data[now_scene_str]
    if now_scene_data.scene_tag == "Cafeteria":
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.NO_IN_RESTAURANT)
def handle_no_in_restaurant(character_id: int) -> int:
    """
    校验角色是否未处于就餐区中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    now_position = character_data.position
    now_scene_str = map_handle.get_map_system_path_str_for_list(now_position)
    now_scene_data = cache.scene_data[now_scene_str]
    if now_scene_data.scene_tag == "Restaurant":
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.HAVE_PLAY_PIANO_IN_SCENE)
def handle_have_play_piano_in_scene(character_id: int) -> int:
    """
    校验场景中是否有人在弹钢琴
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    scene_path = map_handle.get_map_system_path_str_for_list(character_data.position)
    scene_data: game_type.Scene = cache.scene_data[scene_path]
    for now_character_id in scene_data.character_list:
        if now_character_id == character_id:
            continue
        now_character_data: game_type.Character = cache.character_data[character_id]
        if now_character_data.behavior.behavior_id == constant.Behavior.PLAY_PIANO:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.HAVE_SINGING_IN_SCENE)
def handle_have_singing_in_scene(character_id: int) -> int:
    """
    校验场景中是否有人在唱歌
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    scene_path = map_handle.get_map_system_path_str_for_list(character_data.position)
    scene_data: game_type.Scene = cache.scene_data[scene_path]
    for now_character_id in scene_data.character_list:
        if now_character_id == character_id:
            continue
        now_character_data: game_type.Character = cache.character_data[character_id]
        if now_character_data.behavior.behavior_id == constant.Behavior.SINGING:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.PLAYER_NOT_IN_TARGET_SCENE)
def handle_player_not_in_target_scene(character_id: int) -> int:
    """
    校验玩家是否不在角色的目标场景
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    if not character_id:
        return 0
    character_data: game_type.Character = cache.character_data[character_id]
    if not character_data.behavior.move_target:
        return 1
    target_scene_path_str = map_handle.get_map_system_path_str_for_list(character_data.behavior.move_target)
    target_scene = cache.scene_data[target_scene_path_str]
    if 0 in target_scene.character_list:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.PLAYER_IN_TARGET_SCENE)
def handle_player_in_target_scene(character_id: int) -> int:
    """
    校验玩家是否在角色的目标场景
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    if not character_id:
        return 0
    character_data: game_type.Character = cache.character_data[character_id]
    if not character_data.behavior.move_target:
        return 0
    target_scene_path_str = map_handle.get_map_system_path_str_for_list(character_data.behavior.move_target)
    target_scene = cache.scene_data[target_scene_path_str]
    if 0 in target_scene.character_list:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.HAS_NO_CHARACTER_SCENE)
def handle_has_no_character_scene(character_id: int) -> int:
    """
    校验是否存在空无一人的场景
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    if len(cache.no_character_scene_set):
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.NOT_HAS_LIKE_CHARACTER_IN_SCENE)
def handle_not_has_like_character_in_scene(character_id: int) -> int:
    """
    校验场景中是否没有自己喜欢的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(4, set())
    character_data.social_contact.setdefault(5, set())
    if not character_data.social_contact[4] and not character_data.social_contact[5]:
        return 1
    scene_path_str = map_handle.get_map_system_path_str_for_list(character_data.position)
    scene_data: game_type.Scene = cache.scene_data[scene_path_str]
    character_list = []
    for i in {4, 5}:
        for c in character_data.social_contact[i]:
            if c in scene_data.character_list:
                return 0
    return 1

