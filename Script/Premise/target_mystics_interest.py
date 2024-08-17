from Script.Design import handle_premise, attr_calculation, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.TARGET_APOTHECARY_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_apothecary_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否炼金学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 55 in target_data.knowledge_interest:
        if target_data.knowledge_interest[55] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_APOTHECARY_SKILLS_INTEREST_IS_LOW)
def handle_target_apothecary_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否炼金学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 55 in target_data.knowledge_interest:
        if target_data.knowledge_interest[55] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_PARAPSYCHOLOGIES_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_parapsychologies_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否通灵学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 56 in target_data.knowledge_interest:
        if target_data.knowledge_interest[56] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_PARAPSYCHOLOGIES_SKILLS_INTEREST_IS_LOW)
def handle_target_parapsychologies_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否通灵学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 56 in target_data.knowledge_interest:
        if target_data.knowledge_interest[56] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_NUMEROLOGY_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_numerology_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否灵数学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 57 in target_data.knowledge_interest:
        if target_data.knowledge_interest[57] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_NUMEROLOGY_SKILLS_INTEREST_IS_LOW)
def handle_target_numerology_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否灵数学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 57 in target_data.knowledge_interest:
        if target_data.knowledge_interest[57] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_PRACTISE_DIVINATION_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_practise_divination_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否占卜学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 58 in target_data.knowledge_interest:
        if target_data.knowledge_interest[58] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_PRACTISE_DIVINATION_SKILLS_INTEREST_IS_LOW)
def handle_target_practise_divination_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否占卜学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 58 in target_data.knowledge_interest:
        if target_data.knowledge_interest[58] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_PROPHECY_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_prophecy_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否预言学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 59 in target_data.knowledge_interest:
        if target_data.knowledge_interest[59] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_PROPHECY_SKILLS_INTEREST_IS_LOW)
def handle_target_prophecy_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否预言学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 59 in target_data.knowledge_interest:
        if target_data.knowledge_interest[59] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_ASTROLOGY_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_astrology_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否占星学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 60 in target_data.knowledge_interest:
        if target_data.knowledge_interest[60] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_ASTROLOGY_SKILLS_INTEREST_IS_LOW)
def handle_target_astrology_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否占星学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 60 in target_data.knowledge_interest:
        if target_data.knowledge_interest[60] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_DEMONOLOGY_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_demonology_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否恶魔学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 61 in target_data.knowledge_interest:
        if target_data.knowledge_interest[61] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_DEMONOLOGY_SKILLS_INTEREST_IS_LOW)
def handle_target_delonology_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否恶魔学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 61 in target_data.knowledge_interest:
        if target_data.knowledge_interest[61] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_RITUAL_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_ritual_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否仪式学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 62 in target_data.knowledge_interest:
        if target_data.knowledge_interest[62] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_RITUAL_SKILLS_INTEREST_IS_LOW)
def handle_target_ritual_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否仪式学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 62 in target_data.knowledge_interest:
        if target_data.knowledge_interest[62] < 1:
            return 1
        return 0
    return 1
