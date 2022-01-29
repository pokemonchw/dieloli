from Script.Design import handle_premise, attr_calculation
from Script.Core import constant, game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.ETHIC_IS_HEIGHT)
def handle_ethic_is_height(character_id: int) -> int:
    """
    校验角色是否伦理水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[0]
    if 0 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[0])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.ETHIC_IS_LOW)
def handle_ethic_is_low(character_id: int) -> int:
    """
    校验角色是否伦理水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 0 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[0])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.MORALITY_IS_HEIGHT)
def handle_morality_is_height(character_id: int) -> int:
    """
    校验角色是否道德水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[1]
    if 1 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[1])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.MORALITY_IS_LOW)
def handle_morality_is_low(character_id: int) -> int:
    """
    校验角色是否道德水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 1 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[1])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.LITERATURE_IS_HEIGHT)
def handle_literature_is_height(character_id: int) -> int:
    """
    校验角色是否文学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[2]
    if 2 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[2])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.LITERATURE_IS_LOW)
def handle_literature_is_low(character_id: int) -> int:
    """
    校验角色是否文学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 2 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[2])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.POETRY_IS_HEIGHT)
def handle_poetry_is_height(character_id: int) -> int:
    """
    校验角色是否诗歌水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[3]
    if 3 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[3])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.POETRY_IS_LOW)
def handle_poetry_is_low(character_id: int) -> int:
    """
    校验角色是否诗歌水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 3 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[3])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.HISTORY_IS_HEIGHT)
def handle_history_is_height(character_id: int) -> int:
    """
    校验角色是否历史水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[4]
    if 4 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[4])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.HISTORY_IS_LOW)
def handle_history_is_low(character_id: int) -> int:
    """
    校验角色是否历史水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 4 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[4])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.ART_IS_HEIGHT)
def handle_art_is_height(character_id: int) -> int:
    """
    校验角色是否艺术水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[5]
    if 5 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[5])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.ART_IS_LOW)
def handle_art_is_low(character_id: int) -> int:
    """
    校验角色是否艺术水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 5 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[5])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.MUSIC_THEORY_IS_HEIGHT)
def handle_music_theory_is_height(character_id: int) -> int:
    """
    校验角色是否乐理水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[6]
    if 6 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[6])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.MUSIC_THEORY_IS_LOW)
def handle_music_theory_is_low(character_id: int) -> int:
    """
    校验角色是否乐理水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 6 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[6])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.RELIGION_IS_HEIGHT)
def handle_religion_is_height(character_id: int) -> int:
    """
    校验角色是否宗教水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[7]
    if 7 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[7])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.RELIGION_IS_LOW)
def handle_religion_is_low(character_id: int) -> int:
    """
    校验角色是否宗教水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 7 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[7])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.FAITH_IS_HEIGHT)
def handle_faith_is_height(character_id: int) -> int:
    """
    校验角色是否信仰水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[8]
    if 8 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[8])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.FAITH_IS_LOW)
def handle_faith_is_low(character_id: int) -> int:
    """
    校验角色是否信仰水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 8 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[8])
        if level <= 2:
            return 1
        return 0
    return 1
