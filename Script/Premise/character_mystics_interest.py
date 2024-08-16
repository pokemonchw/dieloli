from Script.Design import handle_premise, attr_calculation, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.APOTHECARY_SKILLS_INTEREST_IS_HEIGHT)
def handle_apothecary_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否炼金学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 55 in character_data.knowledge_interest:
        if character_data.knowledge_interest[55] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.APOTHECARY_SKILLS_INTEREST_IS_LOW)
def handle_apothecary_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否炼金学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 55 in character_data.knowledge_interest:
        if character_data.knowledge_interest[55] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.PARAPSYCHOLOGIES_SKILLS_INTEREST_IS_HEIGHT)
def handle_parapsychologies_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否通灵学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 56 in character_data.knowledge_interest:
        if character_data.knowledge_interest[56] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.PARAPSYCHOLOGIES_SKILLS_INTEREST_IS_LOW)
def handle_parapsychologies_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否通灵学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 56 in character_data.knowledge_interest:
        if character_data.knowledge_interest[56] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.NUMEROLOGY_SKILLS_INTEREST_IS_HEIGHT)
def handle_numerology_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否灵数学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 57 in character_data.knowledge_interest:
        if character_data.knowledge_interest[57] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.NUMEROLOGY_SKILLS_INTEREST_IS_LOW)
def handle_numerology_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否灵数学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 57 in character_data.knowledge_interest:
        if character_data.knowledge_interest[57] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.PRACTISE_DIVINATION_SKILLS_INTEREST_IS_HEIGHT)
def handle_practise_divination_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否占卜学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 58 in character_data.knowledge_interest:
        if character_data.knowledge_interest[58] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.PRACTISE_DIVINATION_SKILLS_INTEREST_IS_LOW)
def handle_practise_divination_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否占卜学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 58 in character_data.knowledge_interest:
        if character_data.knowledge_interest[58] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.PROPHECY_SKILLS_INTEREST_IS_HEIGHT)
def handle_prophecy_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否预言学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 59 in character_data.knowledge_interest:
        if character_data.knowledge_interest[59] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.PROPHECY_SKILLS_INTEREST_IS_LOW)
def handle_prophecy_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否预言学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 59 in character_data.knowledge_interest:
        if character_data.knowledge_interest[59] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.ASTROLOGY_SKILLS_INTEREST_IS_HEIGHT)
def handle_astrology_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否占星学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 60 in character_data.knowledge_interest:
        if character_data.knowledge_interest[60] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.ASTROLOGY_SKILLS_INTEREST_IS_LOW)
def handle_astrology_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否占星学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 60 in character_data.knowledge_interest:
        if character_data.knowledge_interest[60] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.DEMONOLOGY_SKILLS_INTEREST_IS_HEIGHT)
def handle_demonology_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否恶魔学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 61 in character_data.knowledge_interest:
        if character_data.knowledge_interest[61] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.DEMONOLOGY_SKILLS_INTEREST_IS_LOW)
def handle_delonology_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否恶魔学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 61 in character_data.knowledge_interest:
        if character_data.knowledge_interest[61] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.RITUAL_SKILLS_INTEREST_IS_HEIGHT)
def handle_ritual_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否仪式学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 62 in character_data.knowledge_interest:
        if character_data.knowledge_interest[62] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.RITUAL_SKILLS_INTEREST_IS_LOW)
def handle_ritual_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否仪式学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 62 in character_data.knowledge_interest:
        if character_data.knowledge_interest[62] < 1:
            return 1
    return 1
