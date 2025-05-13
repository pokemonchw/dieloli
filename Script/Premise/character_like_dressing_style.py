from Script.Design import handle_premise, attr_calculation, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.LIKE_DRESSING_STYLE_IS_CHARMER)
def handle_like_dressing_style_is_charmer(character_id: int) -> int:
    """
    校验角色是否喜欢可爱风格的穿搭
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.like_dressing_style == constant.DressingStyle.STYLE_CHARMER:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.LIKE_DRESSING_STYLE_IS_ALLURES)
def handle_like_dressing_style_is_allures(character_id: int) -> int:
    """
    校验角色是否喜欢性感风格的穿搭
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.like_dressing_style == constant.DressingStyle.STYLE_ALLURES:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.LIKE_DRESSING_STYLE_IS_STYLISH)
def handle_like_dressing_style_is_stylish(character_id: int) -> int:
    """
    校验角色是否喜欢帅气风格的穿搭
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.like_dressing_style == constant.DressingStyle.STYLE_STYLISH:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.LIKE_DRESSING_STYLE_IS_REFRESH)
def handle_like_dressing_style_is_refresh(character_id: int) -> int:
    """
    校验角色是否喜欢清新风格的穿搭
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.like_dressing_style == constant.DressingStyle.STYLE_REFRESH:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.LIKE_DRESSING_STYLE_IS_REFINED)
def handle_like_dressing_style_is_refined(character_id: int) -> int:
    """
    校验角色是否喜欢典雅风格的穿搭
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.like_dressing_style == constant.DressingStyle.STYLE_REFINED:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.LIKE_TARGET_LIKE_DRESSING_STYLE_IS_CHARMER)
def handle_like_target_like_dressing_style_is_charmer(character_id: int) -> int:
    """
    校验角色喜欢的人是否喜欢可爱风格的穿搭
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    like_character_set = set()
    for social_type in {7, 8, 9, 10}:
        if social_type not in character_data.social_contact:
            continue
        like_character_set.update(character_data.social_contact[social_type])
    for now_character_id in like_character_set:
        now_character_data: game_type.Character = cache.character_data[now_character_id]
        if now_character_data.like_dressing_style == constant.DressingStyle.STYLE_CHARMER:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.LIKE_TARGET_LIKE_DRESSING_STYLE_IS_ALLURES)
def handle_like_target_like_dressing_style_is_allures(character_id: int) -> int:
    """
    校验角色喜欢的人是否喜欢性感风格的穿搭
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    like_character_set = set()
    for social_type in {7, 8, 9, 10}:
        if social_type not in character_data.social_contact:
            continue
        like_character_set.update(character_data.social_contact[social_type])
    for now_character_id in like_character_set:
        now_character_data: game_type.Character = cache.character_data[now_character_id]
        if now_character_data.like_dressing_style == constant.DressingStyle.STYLE_ALLURES:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.LIKE_TARGET_LIKE_DRESSING_STYLE_IS_STYLISH)
def handle_like_target_like_dressing_style_is_stylish(character_id: int) -> int:
    """
    校验角色喜欢的人是否喜欢帅气风格的穿搭
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    like_character_set = set()
    for social_type in {7, 8, 9, 10}:
        if social_type not in character_data.social_contact:
            continue
        like_character_set.update(character_data.social_contact[social_type])
    for now_character_id in like_character_set:
        now_character_data: game_type.Character = cache.character_data[now_character_id]
        if now_character_data.like_dressing_style == constant.DressingStyle.STYLE_STYLISH:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.LIKE_TARGET_LIKE_DRESSING_STYLE_IS_REFRESH)
def handle_like_target_like_dressing_style_is_refresh(character_id: int) -> int:
    """
    校验角色喜欢的人是否喜欢清新风格的穿搭
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    like_character_set = set()
    for social_type in {7, 8, 9, 10}:
        if social_type not in character_data.social_contact:
            continue
        like_character_set.update(character_data.social_contact[social_type])
    for now_character_id in like_character_set:
        now_character_data: game_type.Character = cache.character_data[now_character_id]
        if now_character_data.like_dressing_style == constant.DressingStyle.STYLE_REFRESH:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.LIKE_TARGET_LIKE_DRESSING_STYLE_IS_REFINED)
def handle_like_target_like_dressing_style_is_refiend(character_id: int) -> int:
    """
    校验角色喜欢的人是否喜欢典雅风格的穿搭
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    like_character_set = set()
    for social_type in {7, 8, 9, 10}:
        if social_type not in character_data.social_contact:
            continue
        like_character_set.update(character_data.social_contact[social_type])
    for now_character_id in like_character_set:
        now_character_data: game_type.Character = cache.character_data[now_character_id]
        if now_character_data.like_dressing_style == constant.DressingStyle.STYLE_REFINED:
            return 1
    return 0
