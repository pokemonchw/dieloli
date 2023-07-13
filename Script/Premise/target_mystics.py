from Script.Design import handle_premise, attr_calculation, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.TARGET_APOTHECARY_SKILLS_IS_HEIGHT)
def handle_target_apothecary_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否炼金学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 55 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[55])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_APOTHECARY_SKILLS_IS_LOW)
def handle_target_apothecary_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否炼金学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 55 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[55])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_PARAPSYCHOLOGIES_SKILLS_IS_HEIGHT)
def handle_target_parapsychologies_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否通灵学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 56 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[56])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_PARAPSYCHOLOGIES_SKILLS_IS_LOW)
def handle_target_parapsychologies_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否通灵学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 56 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[56])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_NUMEROLOGY_SKILLS_IS_HEIGHT)
def handle_target_numerology_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否灵数学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 57 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[57])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_NUMEROLOGY_SKILLS_IS_LOW)
def handle_target_numerology_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否灵数学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 57 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[57])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_PRACTISE_DIVINATION_SKILLS_IS_HEIGHT)
def handle_target_practise_divination_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否占卜学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 58 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[58])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_PRACTISE_DIVINATION_SKILLS_IS_LOW)
def handle_target_practise_divination_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否占卜学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 58 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[58])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_PROPHECY_SKILLS_IS_HEIGHT)
def handle_target_prophecy_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否预言学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 59 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[59])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_PROPHECY_SKILLS_IS_LOW)
def handle_target_prophecy_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否预言学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 59 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[59])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_ASTROLOGY_SKILLS_IS_HEIGHT)
def handle_target_astrology_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否占星学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 60 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[60])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_ASTROLOGY_SKILLS_IS_LOW)
def handle_target_astrology_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否占星学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 60 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[60])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_DEMONOLOGY_SKILLS_IS_HEIGHT)
def handle_target_demonology_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否恶魔学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 61 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[61])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_DEMONOLOGY_SKILLS_IS_LOW)
def handle_target_delonology_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否恶魔学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 61 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[61])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_RITUAL_SKILLS_IS_HEIGHT)
def handle_target_ritual_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否仪式学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 62 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[62])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_RITUAL_SKILLS_IS_LOW)
def handle_target_ritual_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否仪式学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 62 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[62])
        return level < 3
    return 1
