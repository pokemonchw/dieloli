from Script.Design import handle_premise, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.GET_INTO_PLAYER_SCENE)
def handle_get_into_player_scene(character_id: int) -> int:
    """
    校验角色是否正在进入玩家所在场景
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    now_character_data: game_type.Character = cache.character_data[character_id]
    player_data: game_type.Character = cache.character_data[0]
    if now_character_data.behavior.move_target == player_data.position:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.LEAVE_PLAYER_SCENE)
def handle_leave_player_scene(character_id: int) -> int:
    """
    校验角色是否是从玩家场景离开
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    now_character_data: game_type.Character = cache.character_data[character_id]
    if (
        now_character_data.behavior.move_src == cache.character_data[0].position
        and now_character_data.behavior.move_target != cache.character_data[0].position
        and now_character_data.position != cache.character_data[0].position
    ):
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_IS_SLEEP)
def handle_target_is_sleep(character_id: int) -> int:
    """
    校验交互对象是否正在睡觉
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.state == constant.CharacterStatus.STATUS_SLEEP


@handle_premise.add_premise(constant.Premise.TARGET_IS_SING)
def handle_target_is_sing(character_id: int) -> int:
    """
    校验交互对象是否正在唱歌
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id == -1:
        return 0
    if target_id != character_id:
        target_data: game_type.Character = cache.character_data[target_id]
        return target_data.state == constant.CharacterStatus.STATUS_SINGING
    return 0
