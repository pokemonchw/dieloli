import random
from Script.Design import handle_state_machine, map_handle, constant
from Script.Core import cache_control, game_type

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_state_machine.add_state_machine(constant.StateMachine.CHAT_RAND_CHARACTER)
def character_chat_rand_character(character_id: int):
    """
    角色和场景内随机角色聊天
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    scene_path_str = map_handle.get_map_system_path_str_for_list(character_data.position)
    scene_data: game_type.Scene = cache.scene_data[scene_path_str]
    character_set = scene_data.character_list.copy()
    character_set.remove(character_id)
    character_list = list(character_set)
    target_id = random.choice(character_list)
    character_data.behavior.behavior_id = constant.Behavior.CHAT
    character_data.behavior.duration = 10
    character_data.target_character_id = target_id
    character_data.state = constant.CharacterStatus.STATUS_CHAT


@handle_state_machine.add_state_machine(constant.StateMachine.SING_RAND_CHARACTER)
def character_singing_to_rand_character(character_id: int):
    """
    唱歌给房间里随机角色听
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_list = list(
        cache.scene_data[
            map_handle.get_map_system_path_str_for_list(character_data.position)
        ].character_list
    )
    character_list.remove(character_id)
    target_id = random.choice(character_list)
    character_data.behavior.behavior_id = constant.Behavior.SINGING
    character_data.behavior.duration = 5
    character_data.target_character_id = target_id
    character_data.state = constant.CharacterStatus.STATUS_SINGING


@handle_state_machine.add_state_machine(constant.StateMachine.PLAY_PIANO_RAND_CHARACTER)
def character_play_piano_to_rand_character(character_id: int):
    """
    弹奏钢琴给房间里随机角色听
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_list = list(
        cache.scene_data[
            map_handle.get_map_system_path_str_for_list(character_data.position)
        ].character_list
    )
    character_list.remove(character_id)
    target_id = random.choice(character_list)
    character_data.behavior.behavior_id = constant.Behavior.PLAY_PIANO
    character_data.behavior.duration = 30
    character_data.target_character_id = target_id
    character_data.state = constant.CharacterStatus.STATUS_PLAY_PIANO


@handle_state_machine.add_state_machine(
    constant.StateMachine.TOUCH_HEAD_TO_BEYOND_FRIENDSHIP_TARGET_IN_SCENE
)
def character_touch_head_to_beyond_friendship_target_in_scene(character_id: int):
    """
    对场景中抱有超越友谊想法的随机对象摸头
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    scene_path_str = map_handle.get_map_system_path_str_for_list(character_data.position)
    scene_data: game_type.Scene = cache.scene_data[scene_path_str]
    character_list = []
    for i in {3, 4, 5}:
        character_data.social_contact.setdefault(i, set())
        for c in character_data.social_contact[i]:
            if c in scene_data.character_list:
                character_list.append(c)
    if character_list:
        target_id = random.choice(character_list)
        character_data.behavior.behavior_id = constant.Behavior.TOUCH_HEAD
        character_data.target_character_id = target_id
        character_data.behavior.duration = 2
        character_data.state = constant.CharacterStatus.STATUS_TOUCH_HEAD


@handle_state_machine.add_state_machine(
    constant.StateMachine.EMBRACE_TO_BEYOND_FRIENDSHIP_TARGET_IN_SCENE
)
def character_embrace_to_beyond_friendship_target_in_scene(character_id: int):
    """
    对场景中抱有超越友谊想法的随机对象拥抱
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    scene_path_str = map_handle.get_map_system_path_str_for_list(character_data.position)
    scene_data: game_type.Scene = cache.scene_data[scene_path_str]
    character_list = []
    for i in {3, 4, 5}:
        character_data.social_contact.setdefault(i, set())
        for c in character_data.social_contact[i]:
            if c in scene_data.character_list:
                character_list.append(c)
    if character_list:
        target_id = random.choice(character_list)
        character_data.behavior.behavior_id = constant.Behavior.EMBRACE
        character_data.target_character_id = target_id
        character_data.behavior.duration = 3
        character_data.state = constant.CharacterStatus.STATUS_EMBRACE


@handle_state_machine.add_state_machine(constant.StateMachine.HAND_IN_HAND_TO_LIKE_TARGET_IN_SCENE)
def character_hand_in_hand_to_like_target_in_scene(character_id: int):
    """
    牵住场景中自己喜欢的随机对象的手
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    scene_path_str = map_handle.get_map_system_path_str_for_list(character_data.position)
    scene_data: game_type.Scene = cache.scene_data[scene_path_str]
    character_list = []
    for i in {4, 5}:
        character_data.social_contact.setdefault(i, set())
        for c in character_data.social_contact[i]:
            if c in scene_data.character_list:
                character_list.append(c)
    if character_list:
        target_id = random.choice(character_list)
        character_data.behavior.behavior_id = constant.Behavior.HAND_IN_HAND
        character_data.target_character_id = target_id
        character_data.behavior.duration = 10
        character_data.state = constant.CharacterStatus.STATUS_HAND_IN_HAND


@handle_state_machine.add_state_machine(constant.StateMachine.ABUSE_TO_DISLIKE_TARGET_IN_SCENE)
def character_abuse_to_dislike_target_in_scene(character_id: int):
    """
    辱骂场景中自己讨厌的人
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    scene_path_str = map_handle.get_map_system_path_str_for_list(character_data.position)
    scene_data: game_type.Scene = cache.scene_data[scene_path_str]
    character_list = []
    for i in {0, 1, 2, 3, 4}:
        character_data.social_contact.setdefault(i, set())
        for c in character_data.social_contact[i]:
            if c in scene_data.character_list:
                character_list.append(c)
    if character_list:
        target_id = random.choice(character_list)
        character_data.behavior.behavior_id = constant.Behavior.ABUSE
        character_data.target_character_id = target_id
        character_data.behavior.duration = 10
        character_data.state = constant.CharacterStatus.STATUS_ABUSE
