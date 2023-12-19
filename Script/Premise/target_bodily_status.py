import math
from Script.Design import handle_premise, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.TARGET_HP_IS_HIGHT)
def handle_target_hp_is_hight(character_id: int) -> int:
    """
    校验交互对象是否身体健康
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    hit_point_weight = 0
    if target_data.hit_point:
        hit_point_weight = target_data.hit_point / target_data.hit_point_max
    return hit_point_weight >= 0.75


@handle_premise.add_premise(constant.Premise.TARGET_HP_IS_LOW)
def handle_target_hp_is_low(character_id: int) -> int:
    """
    校验交互对象是否身体不健康
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    hit_point_weight = 0
    if target_data.hit_point:
        hit_point_weight = target_data.hit_point / target_data.hit_point_max
    return hit_point_weight < 0.75


@handle_premise.add_premise(constant.Premise.TARGET_MP_IS_HIGHT)
def handle_target_mp_is_hight(character_id: int) -> int:
    """
    校验交互对象是否身体力充沛
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    mana_point_weight = 0
    if target_data.mana_point:
        mana_point_weight = target_data.mana_point / target_data.mana_point_max
    return mana_point_weight >= 0.5


@handle_premise.add_premise(constant.Premise.TARGET_MP_IS_LOW)
def handle_target_mp_is_low(character_id: int) -> int:
    """
    校验交互对象是否身体力不足
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    mana_point_weight = 0
    if target_data.mana_point:
        mana_point_weight = target_data.mana_point / target_data.mana_point_max
    return mana_point_weight < 0.5


@handle_premise.add_premise(constant.Premise.TARGET_IS_WARM)
def handle_target_is_warm(character_id: int) -> int:
    """
    校验交互对象是否感觉温暖
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    now_warm = 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    for clothing_type in target_data.put_on:
        clothing_id = target_data.put_on[clothing_type]
        if clothing_id is not None and clothing_id != "":
            now_clothing: game_type.Clothing = target_data.clothing[clothing_type][clothing_id]
            now_warm += now_clothing.warm
    return now_warm > 50


@handle_premise.add_premise(constant.Premise.TARGET_NOT_WARM)
def handle_target_not_warm(character_id: int) -> int:
    """
    校验交互对象是否未感觉温暖
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    now_warm = 0
    for clothing_type in target_data.put_on:
        clothing_id = target_data.put_on[clothing_type]
        if clothing_id is not None and clothing_id != "":
            now_clothing: game_type.Clothing = target_data.clothing[clothing_type][clothing_id]
            now_warm += now_clothing.warm
    if now_warm < 50:
        return 0


@handle_premise.add_premise(constant.Premise.TARGET_IS_ACHE)
def handle_target_is_ache(character_id: int) -> int:
    """
    校验交互对象是否感到疼痛
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.status.setdefault(23, 0)
    return target_data.status[23] > 10


@handle_premise.add_premise(constant.Premise.TARGET_NOT_ACHE)
def handle_target_not_ache(character_id: int) -> int:
    """
    校验交互对象是否未感到疼痛
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.status.setdefault(23, 0)
    return target_data.status[23] <= 10


@handle_premise.add_premise(constant.Premise.TARGET_IS_VERTIGO)
def handle_target_is_vetigo(character_id: int) -> int:
    """
    校验交互对象是否感到眩晕
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.status.setdefault(24, 0)
    return target_data.status[24] > 10


@handle_premise.add_premise(constant.Premise.TARGET_NOT_VERTIGO)
def handle_target_not_vertigo(character_id: int) -> int:
    """
    校验交互对象是否未感到眩晕
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.status.setdefault(24, 0)
    return target_data.status[24] <= 10


@handle_premise.add_premise(constant.Premise.TARGET_IS_TIRED)
def handle_target_is_tired(character_id: int) -> int:
    """
    校验交互对象是否感到疲惫
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.status.setdefault(25, 0)
    return target_data.status[25] > 50


@handle_premise.add_premise(constant.Premise.TARGET_NOT_TIRED)
def handle_target_not_tired(character_id: int) -> int:
    """
    校验交互对象是否未感到疲惫
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.status.setdefault(25, 0)
    return target_data.status[25] <= 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_EXTREME_EXHAUSTION)
def handle_target_is_extreme_exhaustion(character_id: int) -> int:
    """
    校验交互对象是否感到极度疲惫
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.status.setdefault(25, 0)
    return target_data.status[25] > 100


@handle_premise.add_premise(constant.Premise.TARGET_IS_INTOXICATED)
def handle_target_is_intoxicated(character_id: int) -> int:
    """
    校验交互对象是否处于迷醉状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.status.setdefault(26, 0)
    return target_data.status[26] > 10


@handle_premise.add_premise(constant.Premise.TARGET_NOT_INTOXICATED)
def handle_target_not_intoxicated(character_id: int) -> int:
    """
    校验交互对象是否未处于迷醉状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.status.setdefault(26, 0)
    return target_data.status[26] <= 10


@handle_premise.add_premise(constant.Premise.TARGET_HUNGER)
def handle_target_hunger(character_id: int) -> int:
    """
    校验交互对象是否处于饥饿状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.status.setdefault(27, 0)
    return target_data.status[27] > 15


@handle_premise.add_premise(constant.Premise.TARGET_NOT_HUNGER)
def handle_target_not_hunger(character_id: int) -> int:
    """
    校验交互对象是否未处于饥饿状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.status.setdefault(27, 0)
    return target_data.status[27] <= 15


@handle_premise.add_premise(constant.Premise.TARGET_THIRSTY)
def handle_target_thirsty(character_id: int) -> int:
    """
    校验交互对象是否处于口渴状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.status.setdefault(28, 0)
    return target_data.status[28] > 15


@handle_premise.add_premise(constant.Premise.TARGET_NOT_THIRSTY)
def handle_target_not_thirsty(character_id: int) -> int:
    """
    校验交互对象是否未处于口渴状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.status.setdefault(28, 0)
    return target_data.status[28] <= 15
