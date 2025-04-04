import random
import numpy
import datetime
from Script.Design import handle_state_machine, character_move, map_handle, constant
from Script.Core import cache_control, game_type

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_state_machine.add_state_machine(constant.StateMachine.MOVE_TO_CLASS)
def character_move_to_classroom(character_id: int):
    """
    移动至所属教室
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    classroom_path = []
    if 0 in character_data.identity_data:
        identity_data: game_type.StudentIdentity = character_data.identity_data[0]
        classroom_path = map_handle.get_map_system_path_for_str(identity_data.classroom)
    elif 1 in character_data.identity_data:
        identity_data: game_type.TeacherIdentity = character_data.identity_data[1]
        classroom_path = map_handle.get_map_system_path_for_str(identity_data.now_classroom)
    if not classroom_path:
        return
    _, path_list, move_path, move_time = character_move.character_move(character_id, classroom_path)
    character_data.behavior.behavior_id = constant.Behavior.MOVE
    character_data.behavior.move_target = move_path
    character_data.behavior.duration = move_time
    character_data.state = constant.CharacterStatus.STATUS_MOVE
    if character_data.follow != -1:
        target_data: game_type.Character = cache.character_data[character_data.follow]
        if target_data.pulling == character_id:
            target_data.pulling = -1
    character_data.follow = -1


@handle_state_machine.add_state_machine(constant.StateMachine.MOVE_TO_OFFICEROOM)
def character_move_to_officeroom(character_id: int):
    """
    移动至所属办公室
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    officeroom_path = []
    if 0 in character_data.identity_data:
        return
    elif 1 in character_data.identity_data:
        identity_data: game_type.TeacherIdentity = character_data.identity_data[1]
        officeroom_path = map_handle.get_map_system_path_for_str(identity_data.officeroom)
    if not classroom_path:
        return
    _, path_list, move_path, move_time = character_move.character_move(character_id, officeroom_path)
    character_data.behavior.behavior_id = constant.Behavior.MOVE
    character_data.behavior.move_target = move_path
    character_data.behavior.duration = move_time
    character_data.state = constant.CharacterStatus.STATUS_MOVE
    if character_data.follow != -1:
        target_data: game_type.Character = cache.character_data[character_data.follow]
        if target_data.pulling == character_id:
            target_data.pulling = -1
    character_data.follow = -1


@handle_state_machine.add_state_machine(constant.StateMachine.MOVE_TO_NEAREST_CAFETERIA)
def character_move_to_rand_cafeteria(character_id: int):
    """
    移动至最近取餐区
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    cafeteria_list = constant.place_data["Cafeteria"]
    now_position_str = map_handle.get_map_system_path_str_for_list(character_data.position)
    time_dict = {}
    for cafeteria in cafeteria_list:
        now_move_time = map_handle.scene_move_time[now_position_str][cafeteria]
        time_dict.setdefault(now_move_time, [])
        time_dict[now_move_time].append(cafeteria)
    min_time = min(time_dict.keys())
    to_cafeteria = map_handle.get_map_system_path_for_str(random.choice(time_dict[min_time]))
    _, _, move_path, move_time = character_move.character_move(character_id, to_cafeteria)
    character_data.behavior.behavior_id = constant.Behavior.MOVE
    character_data.behavior.move_target = move_path
    character_data.behavior.duration = move_time
    character_data.state = constant.CharacterStatus.STATUS_MOVE
    if character_data.follow != -1:
        target_data: game_type.Character = cache.character_data[character_data.follow]
        if target_data.pulling == character_id:
            target_data.pulling = -1
    character_data.follow = -1


@handle_state_machine.add_state_machine(constant.StateMachine.MOVE_TO_NEAREST_RESTAURANT)
def character_move_to_nearest_restaurant(character_id: int):
    """
    设置角色状态为向最近的就餐区移动
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    restaurant_list = constant.place_data["Restaurant"]
    now_position_str = map_handle.get_map_system_path_str_for_list(character_data.position)
    time_dict = {}
    for restaurant in restaurant_list:
        now_move_time = map_handle.scene_move_time[now_position_str][restaurant]
        time_dict.setdefault(now_move_time, [])
        time_dict[now_move_time].append(restaurant)
    min_time = min(time_dict.keys())
    to_restaurant = map_handle.get_map_system_path_for_str(random.choice(time_dict[min_time]))
    _, _, move_path, move_time = character_move.character_move(
        character_id,
        to_restaurant,
    )
    character_data.behavior.behavior_id = constant.Behavior.MOVE
    character_data.behavior.move_target = move_path
    character_data.behavior.duration = move_time
    character_data.state = constant.CharacterStatus.STATUS_MOVE
    if character_data.follow != -1:
        target_data: game_type.Character = cache.character_data[character_data.follow]
        if target_data.pulling == character_id:
            target_data.pulling = -1
    character_data.follow = -1


@handle_state_machine.add_state_machine(constant.StateMachine.MOVE_TO_NEAREST_TOILET)
def character_move_to_nearest_toilet(character_id: int):
    """
    移动至最近的洗手间
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    toilet_list = constant.place_data["Toilet"]
    now_position_str = map_handle.get_map_system_path_str_for_list(character_data.position)
    time_dict = {}
    for toilet in toilet_list:
        now_move_time = map_handle.scene_move_time[now_position_str][toilet]
        time_dict.setdefault(now_move_time, [])
        time_dict[now_move_time].append(toilet)
    min_time = min(time_dict.keys())
    to_toilet = map_handle.get_map_system_path_for_str(random.choice(time_dict[min_time]))
    _, _, move_path, move_time = character_move.character_move(character_id, to_toilet)
    character_data.behavior.behavior_id = constant.Behavior.MOVE
    character_data.behavior.move_target = move_path
    character_data.behavior.duration = move_time
    character_data.state = constant.CharacterStatus.STATUS_MOVE
    if character_data.follow != -1:
        target_data: game_type.Character = cache.character_data[character_data.follow]
        if target_data.pulling == character_id:
            target_data.pulling = -1
    character_data.follow = -1


@handle_state_machine.add_state_machine(constant.StateMachine.MOVE_TO_MUSIC_ROOM)
def character_move_to_music_room(character_id: int):
    """
    移动至音乐活动室
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    to_cafeteria = map_handle.get_map_system_path_for_str(
        random.choice(constant.place_data["MusicClassroom"])
    )
    _, _, move_path, move_time = character_move.character_move(character_id, to_cafeteria)
    character_data.behavior.behavior_id = constant.Behavior.MOVE
    character_data.behavior.move_target = move_path
    character_data.behavior.duration = move_time
    character_data.state = constant.CharacterStatus.STATUS_MOVE
    if character_data.follow != -1:
        target_data: game_type.Character = cache.character_data[character_data.follow]
        if target_data.pulling == character_id:
            target_data.pulling = -1
    character_data.follow = -1


@handle_state_machine.add_state_machine(constant.StateMachine.MOVE_TO_DORMITORY)
def character_move_to_dormitory(character_id: int):
    """
    移动至所在宿舍
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    _, _, move_path, move_time = character_move.character_move(
        character_id,
        map_handle.get_map_system_path_for_str(character_data.dormitory),
    )
    character_data.behavior.behavior_id = constant.Behavior.MOVE
    character_data.behavior.move_target = move_path
    character_data.behavior.duration = move_time
    character_data.state = constant.CharacterStatus.STATUS_MOVE
    if character_data.follow != -1:
        target_data: game_type.Character = cache.character_data[character_data.follow]
        if target_data.pulling == character_id:
            target_data.pulling = -1
    character_data.follow = -1


@handle_state_machine.add_state_machine(constant.StateMachine.MOVE_TO_RAND_SCENE)
def character_move_to_rand_scene(character_id: int):
    """
    移动至随机场景
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    scene_list = list(cache.scene_data.keys())
    now_scene_str = map_handle.get_map_system_path_str_for_list(character_data.position)
    scene_list.remove(now_scene_str)
    target_scene = random.choice(scene_list)
    _, _, move_path, move_time = character_move.character_move(
        character_id,
        map_handle.get_map_system_path_for_str(target_scene),
    )
    character_data.behavior.behavior_id = constant.Behavior.MOVE
    character_data.behavior.move_target = move_path
    character_data.behavior.duration = move_time
    character_data.state = constant.CharacterStatus.STATUS_MOVE
    if character_data.follow != -1:
        target_data: game_type.Character = cache.character_data[character_data.follow]
        if target_data.pulling == character_id:
            target_data.pulling = -1
    character_data.follow = -1


@handle_state_machine.add_state_machine(constant.StateMachine.MOVE_TO_LIKE_TARGET_SCENE)
def character_move_to_like_target_scene(character_id: int):
    """
    移动至随机某个自己喜欢的人所在场景
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_list = []
    for i in {4, 5}:
        character_data.social_contact.setdefault(i, set())
        for c in character_data.social_contact[i]:
            character_list.append(c)
    if character_list:
        target_id = random.choice(character_list)
        target_data: game_type.Character = cache.character_data[target_id]
        _, _, move_path, move_time = character_move.character_move(
            character_id, target_data.position
        )
        character_data.behavior.behavior_id = constant.Behavior.MOVE
        character_data.behavior.move_target = move_path
        character_data.behavior.duration = move_time
        character_data.state = constant.CharacterStatus.STATUS_MOVE
    if character_data.follow != -1:
        target_data: game_type.Character = cache.character_data[character_data.follow]
        if target_data.pulling == character_id:
            target_data.pulling = -1
    character_data.follow = -1


@handle_state_machine.add_state_machine(
    constant.StateMachine.MOVE_TO_NO_FIRST_KISS_LIKE_TARGET_SCENE
)
def character_move_to_no_first_kiss_like_target_scene(character_id: int):
    """
    移动至随机某个自己喜欢的还是初吻的人所在场景
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_list = []
    for i in {4, 5}:
        character_data.social_contact.setdefault(i, set())
        for c in character_data.social_contact[i]:
            c_data: game_type.Character = cache.character_data[c]
            if c_data.first_kiss == -1:
                character_list.append(i)
    if character_list:
        target_id = random.choice(character_list)
        target_data: game_type.Character = cache.character_data[target_id]
        _, _, move_path, move_time = character_move.character_move(
            character_id, target_data.position
        )
        character_data.behavior.behavior_id = constant.Behavior.MOVE
        character_data.behavior.move_target = move_path
        character_data.behavior.duration = move_time
        character_data.state = constant.CharacterStatus.STATUS_MOVE
    if character_data.follow != -1:
        target_data: game_type.Character = cache.character_data[character_data.follow]
        if target_data.pulling == character_id:
            target_data.pulling = -1
    character_data.follow = -1


@handle_state_machine.add_state_machine(
    constant.StateMachine.MOVE_TO_DISLIKE_TARGET_SCENE
)
def character_move_to_dislike_target_scene(character_id: int):
    """
    移动至随机某个自己讨厌的人所在场景
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_list = []
    for i in {0, 1, 2, 3, 4}:
        character_data.social_contact.setdefault(i, set())
        for c in character_data.social_contact[i]:
            character_list.append(c)
    if character_list:
        target_id = random.choice(character_list)
        target_data: game_type.Character = cache.character_data[target_id]
        _, _, move_path, move_time = character_move.character_move(
            character_id, target_data.position
        )
        character_data.behavior.behavior_id = constant.Behavior.MOVE
        character_data.behavior.move_target = move_path
        character_data.behavior.duration = move_time
    if character_data.follow != -1:
        target_data: game_type.Character = cache.character_data[character_data.follow]
        if target_data.pulling == character_id:
            target_data.pulling = -1
    character_data.follow = -1


@handle_state_machine.add_state_machine(
    constant.StateMachine.MOVE_TO_NOT_HAS_DISLIKE_TARGET_SCENE
)
def character_move_to_not_has_dislike_target_scene(character_id: int):
    """
    移动至没有自己讨厌的人的场景
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    scene_set = set()
    for i in {0, 1, 2, 3, 4}:
        character_data.social_contact.setdefault(i, set())
        for c in character_data.social_contact[i]:
            c_data = cache.character_data[c]
            c_position = map_handle.get_map_system_path_str_for_list(c_data.position)
            scene_set.add(c_position)
    if not scene_set:
        return
    now_scene_set = set(cache.scene_data.keys())
    new_scene_list = list(now_scene_set - scene_set)
    if not new_scene_list:
        return
    target_scene_str = random.choice(new_scene_list)
    target_scene = map_handle.get_map_system_path_for_str(target_scene_str)
    _, _, move_path, move_time = character_move.character_move(character_id, target_scene)
    character_data.behavior.behavior_id = constant.Behavior.MOVE
    character_data.behavior.move_target = move_path
    character_data.behavior.duration = move_time
    character_data.state = constant.CharacterStatus.STATUS_MOVE
    if character_data.follow != -1:
        target_data: game_type.Character = cache.character_data[character_data.follow]
        if target_data.pulling == character_id:
            target_data.pulling = -1
    character_data.follow = -1


@handle_state_machine.add_state_machine(constant.StateMachine.MOVE_TO_GROVE)
def character_move_to_grove(character_id: int):
    """
    移动至小树林场景
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    _, _, move_path, move_time = character_move.character_move(character_id, ["7"])
    character_data.behavior.behavior_id = constant.Behavior.MOVE
    character_data.behavior.move_target = move_path
    character_data.behavior.duration = move_time
    character_data.state = constant.CharacterStatus.STATUS_MOVE
    if character_data.follow != -1:
        target_data: game_type.Character = cache.character_data[character_data.follow]
        if target_data.pulling == character_id:
            target_data.pulling = -1
    character_data.follow = -1


@handle_state_machine.add_state_machine(constant.StateMachine.MOVE_TO_ITEM_SHOP)
def character_move_to_item_shop(character_id: int):
    """
    移动至超市场景
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    _, _, move_path, move_time = character_move.character_move(character_id, ["11"])
    character_data.behavior.behavior_id = constant.Behavior.MOVE
    character_data.behavior.move_target = move_path
    character_data.behavior.duration = move_time
    character_data.state = constant.CharacterStatus.STATUS_MOVE
    if character_data.follow != -1:
        target_data: game_type.Character = cache.character_data[character_data.follow]
        if target_data.pulling == character_id:
            target_data.pulling = -1
    character_data.follow = -1


@handle_state_machine.add_state_machine(constant.StateMachine.MOVE_TO_FOLLOW_TARGET_SCENE)
def character_move_to_follow_target_scene(character_id: int):
    """
    角色移动至跟随对象所在场景
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.follow]
    _, _, move_path, move_time = character_move.character_move(character_id, target_data.position)
    character_data.behavior.behavior_id = constant.Behavior.MOVE
    character_data.behavior.move_target = move_path
    character_data.behavior.duration = move_time
    character_data.state = constant.CharacterStatus.STATUS_MOVE


@handle_state_machine.add_state_machine(constant.StateMachine.MOVE_TO_LIBRARY)
def character_move_to_library(character_id: int):
    """
    角色移动至图书馆
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    _, _, move_path, move_time = character_move.character_move(character_id, ["13", "0", "0"])
    character_data.behavior.behavior_id = constant.Behavior.MOVE
    character_data.behavior.move_target = move_path
    character_data.behavior.duration = move_time
    character_data.state = constant.CharacterStatus.STATUS_MOVE
    if character_data.follow != -1:
        target_data: game_type.Character = cache.character_data[character_data.follow]
        if target_data.pulling == character_id:
            target_data.pulling = -1
    character_data.follow = -1


@handle_state_machine.add_state_machine(constant.StateMachine.MOVE_TO_SQUARE)
def character_move_to_square(character_id: int):
    """
    角色移动至操场
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    _, _, move_path, move_time = character_move.character_move(character_id, ["2"])
    character_data.behavior.behavior_id = constant.Behavior.MOVE
    character_data.behavior.move_target = move_path
    character_data.behavior.duration = move_time
    character_data.state = constant.CharacterStatus.STATUS_MOVE
    if character_data.follow != -1:
        target_data: game_type.Character = cache.character_data[character_data.follow]
        if target_data.pulling == character_id:
            target_data.pulling = -1
    character_data.follow = -1

@handle_state_machine.add_state_machine(constant.StateMachine.MOVE_TO_NO_MAN_SCENE)
def character_move_to_no_man_scene(character_id: int):
    """
    角色移动至无人场景
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    now_scene_str = map_handle.get_map_system_path_str_for_list(character_data.position)
    target_scene = cache.no_character_scene_set.pop()
    cache.no_character_scene_set.add(target_scene)
    _, _, move_path, move_time = character_move.character_move(
        character_id,
        map_handle.get_map_system_path_for_str(target_scene)
    )
    character_data.behavior.behavior_id = constant.Behavior.MOVE
    character_data.behavior.move_target = move_path
    character_data.behavior.duration = move_time
    character_data.state = constant.CharacterStatus.STATUS_MOVE
    if character_data.follow != -1:
        target_data: game_type.Character = cache.character_data[character_data.follow]
        if target_data.pulling == character_id:
            target_data.pulling = -1
    character_data.follow = -1


@handle_state_machine.add_state_machine(constant.StateMachine.MOVE_TO_HAVE_CHARACTER_SCENE)
def character_move_to_have_character_scene(character_id: int):
    """
    角色移动至有人的场景
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    scene_set = set(cache.scene_data.keys())
    new_scene_list = list(scene_set - cache.no_character_scene_set)
    target_scene = random.choice(new_scene_list)
    cache.no_character_scene_set.add(target_scene)
    _, _, move_path, move_time = character_move.character_move(
        character_id,
        map_handle.get_map_system_path_for_str(target_scene)
    )
    character_data.behavior.behavior_id = constant.Behavior.MOVE
    character_data.behavior.move_target = move_path
    character_data.behavior.duration = move_time
    character_data.state = constant.CharacterStatus.STATUS_MOVE
    if character_data.follow != -1:
        target_data: game_type.Character = cache.character_data[character_data.follow]
        if target_data.pulling == character_id:
            target_data.pulling = -1
    character_data.follow = -1


@handle_state_machine.add_state_machine(constant.StateMachine.MOVE_TO_CLUB_ACTIVITY_SCENE)
def character_move_to_club_activity_scene(character_id: int):
    """
    角色移动至社团活动场景
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 2 not in character_data.identity_data:
        return
    identity_data: game_type.ClubIdentity = character_data.identity_data[2]
    club_data: game_type.ClubData = cache.all_club_data[identity_data.club_uid]
    now_time = datetime.datetime.fromtimestamp(character_data.behavior.start_time)
    now_week = now_time.weekday()
    if now_week not in club_data.activity_time_dict:
        return
    week_time_dict = club_data.activity_time_dict[now_week]
    now_hour = now_time.hour
    if now_hour not in week_time_dict:
        return
    hour_time_dict = week_time_dict[now_hour]
    now_minute = now_time.minute
    if now_minute not in hour_time_dict:
        return
    activity_id = list(hour_time_dict[now_minute].keys())[0]
    activity_data: game_type.ClubActivityData = club_data.activity_list[activity_id]
    target_scene = activity_data.activity_position
    _, _, move_path, move_time = character_move.character_move(
        character_id,
        target_scene
    )
    character_data.behavior.behavior_id = constant.Behavior.MOVE
    character_data.behavior.move_target = move_path
    character_data.behavior.duration = move_time
    character_data.state = constant.CharacterStatus.STATUS_MOVE
    if character_data.follow != -1:
        target_data: game_type.Character = cache.character_data[character_data.follow]
        if target_data.pulling == character_id:
            target_data.pulling = -1
    character_data.follow = -1


@handle_state_machine.add_state_machine(constant.StateMachine.MOVE_TO_NEAREST_NOT_CLASSROOM)
def character_move_to_nearest_not_classroom(character_id: int):
    """
    角色移动至最近的不是教室的场景
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    now_scene_str = map_handle.get_map_system_path_str_for_list(character_data.position)
    scene_data: game_type.Scene = cache.scene_data[now_scene_str]
    map_path = map_handle.get_map_for_path(character_data.position)
    map_path_str = map_handle.get_map_system_path_str_for_list(map_path)
    map_data = cache.map_data[map_path_str]
    scene_id = character_data.position[-1]
    near_scene_path = ""
    for now_scene_id in map_data.path_edge[scene_id]:
        now_scene_path = map_path + [now_scene_id]
        now_scene_path_str = map_handle.get_map_system_path_str_for_list(now_scene_path)
        now_scene_data = cache.scene_data[now_scene_path_str]
        if not now_scene_data.scene_tag.startswith("Classroom_"):
            near_scene_path = now_scene_path
            break
    _, _, move_path, move_time = character_move.character_move(character_id,near_scene_path)
    character_data.behavior.behavior_id = constant.Behavior.MOVE
    character_data.behavior.move_target = move_path
    character_data.behavior.duration = move_time
    character_data.state = constant.CharacterStatus.STATUS_MOVE
    if character_data.follow != -1:
        target_data: game_type.Character = cache.character_data[character_data.follow]
        if target_data.pulling == character_id:
            target_data.pulling = -1
    character_data.follow = -1


@handle_state_machine.add_state_machine(constant.StateMachine.MOVE_TO_IN_DOOR_SCENE)
def character_move_to_in_door_scene(character_id: int):
    """
    角色移动至室内场景
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    now_scene_str = map_handle.get_map_system_path_str_for_list(character_data.position)
    scene_data: game_type.Scene = cache.scene_data[now_scene_str]
    target_scene_path_str = random.choice(constant.in_door_scene_list)
    target_scene_path = map_handle.get_map_system_path_for_str(target_scene_path_str)
    _, _, move_path, move_time = character_move.character_move(character_id, target_scene_path)
    character_data.behavior.behavior_id = constant.Behavior.MOVE
    character_data.behavior.move_target = move_path
    character_data.behavior.duration = move_time
    character_data.state = constant.CharacterStatus.STATUS_MOVE
    if character_data.follow != -1:
        target_data: game_type.Character = cache.character_data[character_data.follow]
        if target_data.pulling == character_id:
            target_data.pulling = -1
    character_data.follow = -1

