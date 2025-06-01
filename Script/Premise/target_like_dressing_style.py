from Script.Design import handle_premise, attr_calculation, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.TARGET_LIKE_DRESSING_STYLE_IS_CHARMER)
def handle_target_like_dressing_style_is_charmer(character_id: int) -> int:
    """
    校验交互对象是否喜欢可爱风格的穿搭
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_character_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_character_data.like_dressing_style == constant.DressingStyle.STYLE_CHARMER:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_LIKE_DRESSING_STYLE_IS_ALLURES)
def handle_target_like_dressing_style_is_allures(character_id: int) -> int:
    """
    校验交互对象是否喜欢性感风格的穿搭
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_character_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_character_data.like_dressing_style == constant.DressingStyle.STYLE_ALLURES:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_LIKE_DRESSING_STYLE_IS_STYLISH)
def handle_target_like_dressing_style_is_stylish(character_id: int) -> int:
    """
    校验交互对象是否喜欢帅气风格的穿搭
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_character_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_character_data.like_dressing_style == constant.DressingStyle.STYLE_STYLISH:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_LIKE_DRESSING_STYLE_IS_REFRESH)
def handle_target_like_dressing_style_is_refresh(character_id: int) -> int:
    """
    校验交互对象是否喜欢清新风格的穿搭
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_character_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_character_data.like_dressing_style == constant.DressingStyle.STYLE_REFRESH:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_LIKE_DRESSING_STYLE_IS_REFINED)
def handle_target_like_dressing_style_is_refined(character_id: int) -> int:
    """
    校验交互对象是否喜欢典雅风格的穿搭
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_character_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_character_data.like_dressing_style == constant.DressingStyle.STYLE_REFINED:
        return 1
    return 0
