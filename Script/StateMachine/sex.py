import random
from Script.Design import handle_state_machine, map_handle, constant
from Script.Core import cache_control, game_type

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_state_machine.add_state_machine(constant.StateMachine.KISS_TO_LIKE_TARGET_IN_SCENE)
def character_kiss_to_like_target_in_scene(character_id: int):
    """
    和场景中自己喜欢的随机对象接吻
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
        character_data.behavior.behavior_id = constant.Behavior.KISS
        character_data.target_character_id = target_id
        character_data.behavior.duration = 2
        character_data.state = constant.CharacterStatus.STATUS_KISS


@handle_state_machine.add_state_machine(constant.StateMachine.KISS_TO_NO_FIRST_KISS_TARGET_IN_SCENE)
def character_kiss_to_no_first_kiss_like_target_in_scene(character_id: int):
    """
    和场景中自己喜欢的还是初吻的随机对象接吻
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
                c_data: game_type.Character = cache.character_data[c]
                if c_data.first_kiss == -1:
                    character_list.append(c)
    if character_list:
        target_id = random.choice(character_list)
        character_data.behavior.behavior_id = constant.Behavior.KISS
        character_data.target_character_id = target_id
        character_data.behavior.duration = 2
        character_data.state = constant.CharacterStatus.STATUS_KISS


@handle_state_machine.add_state_machine(constant.StateMachine.MASTURBATION)
def character_masturbation(character_id: int):
    """
    角色手淫
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.behavior.behavior_id = constant.Behavior.MASTURBATION
    character_data.behavior.duration = 10
    character_data.state = constant.CharacterStatus.STATUS_MASTURBATION


@handle_state_machine.add_state_machine(constant.StateMachine.MISSIONARY_POSITION)
def character_missionary_position(character_id: int):
    """
    和交互对象正常体位做爱
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    add_time = character_data.behavior.start_time - target_data.behavior.start_time
    if add_time <= 0:
        add_time = 1
    constant.settle_behavior_effect_data[constant.BehaviorEffect.INTERRUPT_TARGET_ACTIVITY](character_id,1,None,0)
    target_data.behavior.start_time = character_data.behavior.start_time
    if not character_data.sex:
        character_data.behavior.behavior_id = constant.Behavior.MISSIONARY_POSITION_MALE
        character_data.state = constant.CharacterStatus.STATUS_MISSIONARY_POSITION_MALE
        target_data.behavior.behavior_id = constant.Behavior.MISSIONARY_POSITION_FEMALE
        target_data.state = constant.CharacterStatus.STATUS_MISSIONARY_POSITION_FEMALE
    else:
        target_data.behavior.behavior_id = constant.Behavior.MISSIONARY_POSITION_MALE
        target_data.state = constant.CharacterStatus.STATUS_MISSIONARY_POSITION_MALE
        character_data.behavior.behavior_id = constant.Behavior.MISSIONARY_POSITION_FEMALE
        character_data.state = constant.CharacterStatus.STATUS_MISSIONARY_POSITION_FEMALE
    target_data.behavior.duration = 10
    character_data.behavior.duration = 10
