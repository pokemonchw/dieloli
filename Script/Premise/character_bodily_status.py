import math
from Script.Design import handle_premise, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.HP_IS_HIGHT)
def handle_hp_is_hight(character_id: int) -> int:
    """
    校验角色是否身体健康
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    hit_point_weight = 0
    if character_data.hit_point:
        hit_point_weight = character_data.hit_point / character_data.hit_point_max
    if hit_point_weight >= 0.75:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.HP_IS_LOW)
def handle_hp_is_low(character_id: int) -> int:
    """
    校验角色是否身体不健康
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    hit_point_weight = 0
    if character_data.hit_point:
        hit_point_weight = character_data.hit_point / character_data.hit_point_max
    if hit_point_weight < 0.75:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.MP_IS_HIGHT)
def handle_mp_is_hight(character_id: int) -> int:
    """
    校验角色是否体力充沛
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    mana_point_weight = 0
    if character_data.mana_point:
        mana_point_weight = character_data.mana_point / character_data.mana_point_max
    if mana_point_weight >= 0.5:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.MP_IS_LOW)
def handle_mp_is_low(character_id: int) -> int:
    """
    校验角色是否体力不足
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    mana_point_weight = 0
    if character_data.mana_point:
        mana_point_weight = character_data.mana_point / character_data.mana_point_max
    if mana_point_weight < 0.5:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_WARM)
def handle_is_warm(character_id: int) -> int:
    """
    校验角色是否感觉温暖
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    now_warm = 0
    for clothing_type in character_data.put_on:
        clothing_id = character_data.put_on[clothing_type]
        if clothing_id is not None and clothing_id != "":
            now_clothing: game_type.Clothing = character_data.clothing[clothing_type][clothing_id]
            now_warm += now_clothing.warm
    if now_warm > 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.NOT_WARM)
def handle_not_warm(character_id: int) -> int:
    """
    校验角色是否未感觉温暖
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    now_warm = 0
    for clothing_type in character_data.put_on:
        clothing_id = character_data.put_on[clothing_type]
        if clothing_id is not None and clothing_id != "":
            now_clothing: game_type.Clothing = character_data.clothing[clothing_type][clothing_id]
            now_warm += now_clothing.warm
    if now_warm < 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_ACHE)
def handle_is_ache(character_id: int) -> int:
    """
    校验角色是否感到疼痛
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(23, 0)
    if character_data.status[23] > 10:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.NOT_ACHE)
def handle_not_ache(character_id: int) -> int:
    """
    校验角色是否未感到疼痛
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(23, 0)
    if character_data.status[23] <= 10:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_VERTIGO)
def handle_is_vetigo(character_id: int) -> int:
    """
    校验角色是否感到眩晕
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(24, 0)
    if character_data.status[24] > 10:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.NOT_VERTIGO)
def handle_not_vertigo(character_id: int) -> int:
    """
    校验角色是否未感到眩晕
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(24, 0)
    if character_data.status[24] <= 10:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_TIRED)
def handle_is_tired(character_id: int) -> int:
    """
    校验角色是否感到疲惫
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(25, 0)
    if character_data.status[25] > 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.NOT_TIRED)
def handle_not_tired(character_id: int) -> int:
    """
    校验角色是否未感到疲惫
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(25, 0)
    if character_data.status[25] <= 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_EXTREME_EXHAUSTION)
def handle_is_extreme_exhaustion(character_id: int) -> int:
    """
    校验角色是否感到极度疲惫
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(25, 0)
    if character_data.status[25] > 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_INTOXICATED)
def handle_is_intoxicated(character_id: int) -> int:
    """
    校验角色是否处于迷醉状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(26, 0)
    if character_data.status[26] > 10:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.NOT_INTOXICATED)
def handle_not_intoxicated(character_id: int) -> int:
    """
    校验角色是否未处于迷醉状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(26, 0)
    if character_data.status[26] <= 10:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.HUNGER)
def handle_hunger(character_id: int) -> int:
    """
    校验角色是否处于饥饿状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(27, 0)
    if character_data.status[27] > 15:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.NOT_HUNGER)
def handle_not_hunger(character_id: int) -> int:
    """
    校验角色是否未处于饥饿状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(27, 0)
    if character_data.status[27] <= 15:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.THIRSTY)
def handle_thirsty(character_id: int) -> int:
    """
    校验角色是否处于口渴状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(28, 0)
    if character_data.status[28] > 15:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.NOT_THIRSTY)
def handle_not_thirsty(character_id: int) -> int:
    """
    校验角色是否未处于口渴状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(28, 0)
    if character_data.status[28] <= 15:
        return 1
    return 0
