from Script.Design import handle_premise, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.IS_FOLLOW_PLAYER)
def handle_is_follow_player(character_id: int) -> int:
    """
    校验角色是否正在跟随玩家
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.follow == 0:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.HAVE_FOLLOW)
def handle_have_follow(character_id: int) -> int:
    """
    校验角色是否拥有跟随对象
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.follow != -1 and character_data.follow != character_id:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.NO_IN_FOLLOW_TARGET_SCENE)
def handle_no_in_follow_target_scene(character_id: int) -> int:
    """
    判断角色是否不在跟随的对象的场景
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.follow == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.follow]
    if character_data.position != target_data.position:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_IS_FOLLOW_PLAYER)
def handle_target_is_follow_player(character_id: int) -> int:
    """
    校验交互对象是否正在跟随玩家
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id != -1 and character_data.target_character_id != character_id:
        target_data: game_type.Character = cache.character_data[character_data.target_character_id]
        if target_data.follow == 0:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_NOT_FOLLOW_PLAYER)
def handle_target_not_follow_player(character_id: int) -> int:
    """
    校验交互对象是否未跟随玩家
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id in {-1, 0}:
        return 1
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.follow:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_COLLECTION_SYSTEM)
def handle_is_collection_system(character_id: int) -> int:
    """
    校验玩家是否已启用收藏模式
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    if cache.is_collection:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.UN_COLLECTION_SYSTEM)
def handle_un_collection_system(character_id: int) -> int:
    """
    校验玩家是否未启用收藏模式
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    if not cache.is_collection:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_IS_COLLECTION)
def handle_target_is_collection(character_id: int) -> int:
    """
    校验交互对象是否已被玩家收藏
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    player_data: game_type.Character = cache.character_data[0]
    if character_data.target_character_id in player_data.collection_character:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_IS_NOT_COLLECTION)
def handle_target_is_not_collection(character_id: int) -> int:
    """
    校验交互对象是否未被玩家收藏
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    player_data: game_type.Character = cache.character_data[0]
    if character_data.target_character_id not in player_data.collection_character:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.OBSERVE_IS_ON)
def handle_observe_is_on(character_id: int) -> int:
    """
    校验是否已开启看海模式
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    if cache.observe_switch:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.OBSERVE_IS_OFF)
def handle_observe_is_off(character_id: int) -> int:
    """
    校验是否已关闭看海模式
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    if cache.observe_switch:
        return 0
    return 1
