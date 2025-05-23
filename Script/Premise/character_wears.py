from uuid import UUID
from Script.Design import handle_premise, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


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


@handle_premise.add_premise(constant.Premise.DRESSING_STYLE_IS_CHARMER)
def handle_dressing_style_is_charmer(character_id: int) -> int:
    """
    校验角色是否穿着风格为可爱
    Keyword arguments:
    character_id -- 角色id
    Return argument:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    all_sexy = 0
    all_handsome = 0
    all_elegant = 0
    all_fresh = 0
    all_sweet = 0
    for clothing_type_id in character_data.put_on:
        clothing_id = character_data.put_on[clothing_type_id]
        if clothing_id not in {None, ""}:
            clothing_data = character_data.clothing[clothing_type_id][clothing_id]
            all_sexy += clothing_data.sexy
            all_handsome += clothing_data.handsome
            all_elegant += clothing_data.elegant
            all_fresh += clothing_data.fresh
            all_sweet += clothing_data.sweet
    if all_sweet == max([all_sexy, all_handsome, all_elegant, all_fresh, all_sweet]):
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.DRESSING_STYLE_IS_ALLURES)
def handle_dressing_style_is_allures(character_id: int) -> int:
    """
    校验角色是否穿着风格为性感
    Keyword arguments:
    character_id -- 角色id
    Return argument:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    all_sexy = 0
    all_handsome = 0
    all_elegant = 0
    all_fresh = 0
    all_sweet = 0
    for clothing_type_id in character_data.put_on:
        clothing_id = character_data.put_on[clothing_type_id]
        if clothing_id not in {None, ""}:
            clothing_data = character_data.clothing[clothing_type_id][clothing_id]
            all_sexy += clothing_data.sexy
            all_handsome += clothing_data.handsome
            all_elegant += clothing_data.elegant
            all_fresh += clothing_data.fresh
            all_sweet += clothing_data.sweet
    if all_sexy == max([all_sexy, all_handsome, all_elegant, all_fresh, all_sweet]):
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.DRESSING_STYLE_IS_STYLISH)
def handle_dressing_style_is_stylish(character_id: int) -> int:
    """
    校验角色是否穿着风格为帅气
    Keyword arguments:
    character_id -- 角色id
    Return argument:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    all_sexy = 0
    all_handsome = 0
    all_elegant = 0
    all_fresh = 0
    all_sweet = 0
    for clothing_type_id in character_data.put_on:
        clothing_id = character_data.put_on[clothing_type_id]
        if clothing_id not in {None, ""}:
            clothing_data = character_data.clothing[clothing_type_id][clothing_id]
            all_sexy += clothing_data.sexy
            all_handsome += clothing_data.handsome
            all_elegant += clothing_data.elegant
            all_fresh += clothing_data.fresh
            all_sweet += clothing_data.sweet
    if all_handsome == max([all_sexy, all_handsome, all_elegant, all_fresh, all_sweet]):
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.DRESSING_STYLE_IS_REFRESH)
def handle_dressing_style_is_refresh(character_id: int) -> int:
    """
    校验角色是否穿着风格为清新
    Keyword arguments:
    character_id -- 角色id
    Return argument:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    all_sexy = 0
    all_handsome = 0
    all_elegant = 0
    all_fresh = 0
    all_sweet = 0
    for clothing_type_id in character_data.put_on:
        clothing_id = character_data.put_on[clothing_type_id]
        if clothing_id not in {None, ""}:
            clothing_data = character_data.clothing[clothing_type_id][clothing_id]
            all_sexy += clothing_data.sexy
            all_handsome += clothing_data.handsome
            all_elegant += clothing_data.elegant
            all_fresh += clothing_data.fresh
            all_sweet += clothing_data.sweet
    if all_fresh == max([all_sexy, all_handsome, all_elegant, all_fresh, all_sweet]):
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.DRESSING_STYLE_IS_REFINED)
def handle_dressing_style_is_refined(character_id: int) -> int:
    """
    校验角色是否穿着风格为典雅
    Keyword arguments:
    character_id -- 角色id
    Return argument:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    all_sexy = 0
    all_handsome = 0
    all_elegant = 0
    all_fresh = 0
    all_sweet = 0
    for clothing_type_id in character_data.put_on:
        clothing_id = character_data.put_on[clothing_type_id]
        if clothing_id not in {None, ""}:
            clothing_data = character_data.clothing[clothing_type_id][clothing_id]
            all_sexy += clothing_data.sexy
            all_handsome += clothing_data.handsome
            all_elegant += clothing_data.elegant
            all_fresh += clothing_data.fresh
            all_sweet += clothing_data.sweet
    if all_elegant == max([all_sexy, all_handsome, all_elegant, all_fresh, all_sweet]):
        return 1
    return 0
