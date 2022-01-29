from Script.Design import handle_premise, attr_calculation
from Script.Core import constant, game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.APOTHECARY_SKILLS_IS_HEIGHT)
def handle_apothecary_skills_is_height(character_id: int) -> int:
    """
    校验角色是否炼金学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[55]
    if 55 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[55])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.APOTHECARY_SKILLS_IS_LOW)
def handle_apothecary_skills_is_low(character_id: int) -> int:
    """
    校验角色是否炼金学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 55 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[55])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.PARAPSYCHOLOGIES_SKILLS_IS_HEIGHT)
def handle_parapsychologies_skills_is_height(character_id: int) -> int:
    """
    校验角色是否通灵学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[56]
    if 56 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[56])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.PARAPSYCHOLOGIES_SKILLS_IS_LOW)
def handle_parapsychologies_skills_is_low(character_id: int) -> int:
    """
    校验角色是否通灵学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 56 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[56])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.NUMEROLOGY_SKILLS_IS_HEIGHT)
def handle_numerology_skills_is_height(character_id: int) -> int:
    """
    校验角色是否灵数学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[57]
    if 57 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[57])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.NUMEROLOGY_SKILLS_IS_LOW)
def handle_numerology_skills_is_low(character_id: int) -> int:
    """
    校验角色是否灵数学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 57 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[57])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.PRACTISE_DIVINATION_SKILLS_IS_HEIGHT)
def handle_practise_divination_skills_is_height(character_id: int) -> int:
    """
    校验角色是否占卜学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[58]
    if 58 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[58])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.PRACTISE_DIVINATION_SKILLS_IS_LOW)
def handle_practise_divination_skills_is_low(character_id: int) -> int:
    """
    校验角色是否占卜学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 58 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[58])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.PROPHECY_SKILLS_IS_HEIGHT)
def handle_prophecy_skills_is_height(character_id: int) -> int:
    """
    校验角色是否预言学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[59]
    if 59 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[59])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.PROPHECY_SKILLS_IS_LOW)
def handle_prophecy_skills_is_low(character_id: int) -> int:
    """
    校验角色是否预言学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 59 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[59])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.ASTROLOGY_SKILLS_IS_HEIGHT)
def handle_astrology_skills_is_height(character_id: int) -> int:
    """
    校验角色是否占星学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[60]
    if 60 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[60])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.ASTROLOGY_SKILLS_IS_LOW)
def handle_astrology_skills_is_low(character_id: int) -> int:
    """
    校验角色是否占星学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 60 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[60])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.DEMONOLOGY_SKILLS_IS_HEIGHT)
def handle_demonology_skills_is_height(character_id: int) -> int:
    """
    校验角色是否恶魔学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[61]
    if 61 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[61])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.DEMONOLOGY_SKILLS_IS_LOW)
def handle_delonology_skills_is_low(character_id: int) -> int:
    """
    校验角色是否恶魔学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 61 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[61])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.RITUAL_SKILLS_IS_HEIGHT)
def handle_ritual_skills_is_height(character_id: int) -> int:
    """
    校验角色是否仪式学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[62]
    if 62 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[62])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.RITUAL_SKILLS_IS_LOW)
def handle_ritual_skills_is_low(character_id: int) -> int:
    """
    校验角色是否仪式学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 62 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[62])
        if level <= 2:
            return 1
        return 0
    return 1
