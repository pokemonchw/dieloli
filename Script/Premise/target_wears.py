from uuid import UUID
from Script.Design import handle_premise, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.TARGET_IS_NAKED)
def handle_target_is_naked(character_id: int) -> int:
    """
    校验交互对象是否一丝不挂
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    for i in target_data.put_on:
        if isinstance(target_data.put_on[i], UUID):
            return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_NOT_PUT_ON_UNDERWEAR)
def handle_target_not_put_underwear(character_id: int) -> int:
    """
    校验角色的目标对象是否没穿上衣
    Keyword arguments:
    character_id -- 角色id
    Return argument:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data = cache.character_data[character_data.target_character_id]
    if (1 not in target_data.put_on) or (target_data.put_on[1] == ""):
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_NOT_PUT_ON_SKIRT)
def handle_target_not_put_on_skirt(character_id: int) -> int:
    """
    校验角色的目标对象是否没穿短裙
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data = cache.character_data[character_data.target_character_id]
    if (3 not in target_data.put_on) or (target_data.put_on[3] == ""):
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_NOT_PUT_ON_BRA)
def handle_target_not_put_on_bra(character_id: int) -> int:
    """
    校验交互对象是否没穿胸罩
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data = cache.character_data[character_data.target_character_id]
    if (6 not in target_data.put_on) or (target_data.put_on[6] == ""):
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_NOT_PUT_ON_COAT)
def handle_target_not_put_on_coat(character_id: int) -> int:
    """
    校验交互对象是否没穿外套
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data = cache.character_data[character_data.target_character_id]
    if (0 not in target_data.put_on) or (target_data.put_on[0] == ""):
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_NOT_PUT_ON_PANTS)
def handle_target_not_put_on_pants(character_id: int) -> int:
    """
    校验交互对象是否没穿裤子
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data = cache.character_data[character_data.target_character_id]
    if (2 not in target_data.put_on) or (target_data.put_on[2] == ""):
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_NOT_PUT_ON_UNDERPANTS)
def handle_target_not_put_on_underpants(character_id: int) -> int:
    """
    校验交互对象是否没穿内裤
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data = cache.character_data[character_data.target_character_id]
    if (7 not in target_data.put_on) or (target_data.put_on[7] == ""):
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_NOT_NAKED)
def handle_target_not_naked(character_id: int) -> int:
    """
    校验交互对象是否不是一丝不挂
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    for i in target_data.put_on:
        if isinstance(target_data.put_on[i], UUID):
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_WEAR_COAT)
def handle_target_wear_coat(character_id: int) -> int:
    """
    校验交互对象是否穿着外套
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 0 not in target_data.put_on:
        return 0
    if target_data.put_on[0] == "":
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_WEAR_UNDERWEAR)
def handle_target_wear_underwear(character_id: int) -> int:
    """
    校验交互对象是否穿着上衣
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 1 not in target_data.put_on:
        return 0
    if target_data.put_on[1] == "":
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_WEAR_UNDERPANTS)
def handle_target_wear_underpants(character_id: int) -> int:
    """
    校验交互对象是否穿着内裤
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 7 not in target_data.put_on:
        return 0
    if target_data.put_on[7] == "":
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_WEAR_BRA)
def handle_target_wear_bra(character_id: int) -> int:
    """
    校验交互对象是否穿着胸罩
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 6 not in target_data.put_on:
        return 0
    if target_data.put_on[6] == "":
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_WEAR_PANTS)
def handle_target_wear_pants(character_id: int) -> int:
    """
    校验交互对象是否穿着裤子
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 2 not in target_data.put_on:
        return 0
    if target_data.put_on[2] == "":
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_WEAR_SKIRT)
def handle_target_wear_skirt(character_id: int) -> int:
    """
    校验交互对象是否穿着短裙
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 3 not in target_data.put_on:
        return 0
    if target_data.put_on[3] == "":
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_WEAR_SHOES)
def handle_target_wear_shoes(character_id: int) -> int:
    """
    校验交互对象是否穿着鞋子
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 4 not in target_data.put_on:
        return 0
    if target_data.put_on[4] == "":
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_WEAR_SOCKS)
def handle_target_wear_socks(character_id: int) -> int:
    """
    校验交互对象是否穿着袜子
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 5 not in target_data.put_on:
        return 0
    if target_data.put_on[5] == "":
        return 0
    return 1
