from uuid import UUID
from Script.Design import handle_premise, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


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
def handle_target_put_on_skirt(character_id: int) -> int:
    """
    校验角色的目标对象是否穿着短裙
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
        return 0
    return 1


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


@handle_premise.add_premise(constant.Premise.NO_WEAR_UNDERWEAR)
def handle_no_wear_underwear(character_id: int) -> int:
    """
    校验角色是否没穿上衣
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if (
        1 not in character_data.put_on
        or character_data.put_on[1] is None
        or character_data.put_on[1] == ""
    ):
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.NO_WEAR_UNDERPANTS)
def handle_no_wear_underpants(character_id: int) -> int:
    """
    校验角色是否没穿内裤
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if (
        7 not in character_data.put_on
        or character_data.put_on[7] is None
        or character_data.put_on[7] == ""
    ):
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.NO_WEAR_BRA)
def handle_no_wear_bra(character_id: int) -> int:
    """
    校验角色是否没穿胸罩
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if (
        6 not in character_data.put_on
        or character_data.put_on[6] is None
        or character_data.put_on[6] == ""
    ):
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.NO_WEAR_PANTS)
def handle_no_wear_pants(character_id: int) -> int:
    """
    校验角色是否没穿裤子
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if (
        2 not in character_data.put_on
        or character_data.put_on[2] is None
        or character_data.put_on[2] == ""
    ):
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.NO_WEAR_SKIRT)
def handle_no_wear_skirt(character_id: int) -> int:
    """
    校验角色是否没穿短裙
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if (
        3 not in character_data.put_on
        or character_data.put_on[3] is None
        or character_data.put_on[3] == ""
    ):
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.NO_WEAR_SHOES)
def handle_no_wear_shoes(character_id: int) -> int:
    """
    校验角色是否没穿鞋子
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if (
        4 not in character_data.put_on
        or character_data.put_on[4] is None
        or character_data.put_on[4] == ""
    ):
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.NO_WEAR_SOCKS)
def handle_no_wear_socks(character_id: int) -> int:
    """
    校验角色是否没穿袜子
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if (
        5 not in character_data.put_on
        or character_data.put_on[5] is None
        or character_data.put_on[5] == ""
    ):
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.PLAYER_IS_NAKED)
def handle_player_is_naked(character_id: int) -> int:
    """
    校验玩家是否一丝不挂
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    player_data: game_type.Character = cache.character_data[character_id]
    for i in player_data.put_on:
        if isinstance(player_data.put_on[i],UUID):
            return 0
    return 1


@handle_premise.add_premise(constant.Premise.IS_NAKED)
def handle_is_naked(character_id: int) -> int:
    """
    校验角色是否一丝不挂
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    for i in character_data.put_on:
        if isinstance(character_data.put_on[i], UUID):
            return 0
    return 1


@handle_premise.add_premise(constant.Premise.NOT_NAKED)
def handle_not_naked(character_id: int) -> int:
    """
    校验角色是否不是一丝不挂
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    for i in character_data.put_on:
        if isinstance(character_data.put_on[i], UUID):
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.NO_WEAR_COAT)
def handle_no_wear_coat(character_id: int) -> int:
    """
    校验角色是否没穿外套
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if (
        0 not in character_data.put_on
        or character_data.put_on[0] is None
        or character_data.put_on[0] == ""
    ):
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.WEAR_COAT)
def handle_wear_coat(character_id: int) -> int:
    """
    校验角色是否穿了外套
    Keyword arguments:
    character_id -- 角色id
    Return argument:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if (
        0 not in character_data.put_on
        or character_data.put_on[0] is None
        or character_data.put_on[0] == ""
    ):
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.WEAR_UNDERWEAR)
def handle_wear_underwear(character_id: int) -> int:
    """
    校验角色是否穿了上衣
    Keyword arguments:
    character_id -- 角色id
    Return argument:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if (
        1 not in character_data.put_on
        or character_data.put_on[1] is None
        or character_data.put_on[1] == ""
    ):
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.WEAR_UNDERPANTS)
def handle_wear_underpants(character_id: int) -> int:
    """
    校验角色是否穿了内裤
    Keyword arguments:
    character_id -- 角色id
    Return argument:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if (
        7 not in character_data.put_on
        or character_data.put_on[7] is None
        or character_data.put_on[7] == ""
    ):
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.WEAR_BRA)
def handle_wear_bra(character_id: int) -> int:
    """
    校验角色是否穿了胸罩
    Keyword arguments:
    character_id -- 角色id
    Return argument:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if (
        6 not in character_data.put_on
        or character_data.put_on[6] is None
        or character_data.put_on[6] == ""
    ):
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.WEAR_PANTS)
def handle_wear_pants(character_id: int) -> int:
    """
    校验角色是否穿了裤子
    Keyword arguments:
    character_id -- 角色id
    Return argument:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if (
        2 not in character_data.put_on
        or character_data.put_on[2] is None
        or character_data.put_on[2] == ""
    ):
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.WEAR_SKIRT)
def handle_wear_skirt(character_id: int) -> int:
    """
    校验角色是否穿了短裙
    Keyword arguments:
    character_id -- 角色id
    Return argument:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if (
        3 not in character_data.put_on
        or character_data.put_on[3] is None
        or character_data.put_on[3] == ""
    ):
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.WEAR_SHOES)
def handle_wear_shoes(character_id: int) -> int:
    """
    校验角色是否穿了鞋子
    Keyword arguments:
    character_id -- 角色id
    Return argument:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if (
        4 not in character_data.put_on
        or character_data.put_on[4] is None
        or character_data.put_on[4] == ""
    ):
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.WEAR_SOCKS)
def handle_wear_socks(character_id: int) -> int:
    """
    校验角色是否穿了袜子
    Keyword arguments:
    character_id -- 角色id
    Return argument:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if (
        5 not in character_data.put_on
        or character_data.put_on[5] is None
        or character_data.put_on[5] == ""
    ):
        return 0
    return 1
