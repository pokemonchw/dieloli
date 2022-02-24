from Script.Design import handle_premise, attr_calculation
from Script.Core import constant, game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.TARGET_ETHIC_IS_HEIGHT)
def handle_target_ethic_is_height(character_id: int) -> int:
    """
    校验交互对象是否伦理水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    weight = 1 + target_data.knowledge_interest[0]
    if 0 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[0])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.TARGET_ETHIC_IS_LOW)
def handle_target_ethic_is_low(character_id: int) -> int:
    """
    校验交互对象是否伦理水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 0 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[0])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_MORALITY_IS_HEIGHT)
def handle_target_morality_is_height(character_id: int) -> int:
    """
    校验交互对象是否道德水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    weight = 1 + target_data.knowledge_interest[1]
    if 1 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[1])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.TARGET_MORALITY_IS_LOW)
def handle_target_morality_is_low(character_id: int) -> int:
    """
    校验交互对象是否道德水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 1 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[1])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_LITERATURE_IS_HEIGHT)
def handle_target_literature_is_height(character_id: int) -> int:
    """
    校验交互对象是否文学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    weight = 1 + target_data.knowledge_interest[2]
    if 2 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[2])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.TARGET_LITERATURE_IS_LOW)
def handle_target_literature_is_low(character_id: int) -> int:
    """
    校验交互对象是否文学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 2 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[2])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_POETRY_IS_HEIGHT)
def handle_target_poetry_is_height(character_id: int) -> int:
    """
    校验交互对象是否诗歌水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    weight = 1 + target_data.knowledge_interest[3]
    if 3 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[3])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.TARGET_POETRY_IS_LOW)
def handle_target_poetry_is_low(character_id: int) -> int:
    """
    校验交互对象是否诗歌水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 3 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[3])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_HISTORY_IS_HEIGHT)
def handle_target_history_is_height(character_id: int) -> int:
    """
    校验交互对象是否历史水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    weight = 1 + target_data.knowledge_interest[4]
    if 4 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[4])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.TARGET_HISTORY_IS_LOW)
def handle_target_history_is_low(character_id: int) -> int:
    """
    校验交互对象是否历史水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 4 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[4])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_ART_IS_HEIGHT)
def handle_target_art_is_height(character_id: int) -> int:
    """
    校验交互对象是否艺术水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    weight = 1 + target_data.knowledge_interest[5]
    if 5 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[5])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.TARGET_ART_IS_LOW)
def handle_target_art_is_low(character_id: int) -> int:
    """
    校验交互对象是否艺术水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 5 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[5])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_MUSIC_THEORY_IS_HEIGHT)
def handle_target_music_theory_is_height(character_id: int) -> int:
    """
    校验交互对象是否乐理水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    weight = 1 + target_data.knowledge_interest[6]
    if 6 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[6])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.TARGET_MUSIC_THEORY_IS_LOW)
def handle_target_music_theory_is_low(character_id: int) -> int:
    """
    校验交互对象是否乐理水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 6 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[6])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_RELIGION_IS_HEIGHT)
def handle_target_religion_is_height(character_id: int) -> int:
    """
    校验交互对象是否宗教水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    weight = 1 + target_data.knowledge_interest[7]
    if 7 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[7])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.TARGET_RELIGION_IS_LOW)
def handle_target_religion_is_low(character_id: int) -> int:
    """
    校验交互对象是否宗教水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 7 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[7])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_FAITH_IS_HEIGHT)
def handle_target_faith_is_height(character_id: int) -> int:
    """
    校验交互对象是否信仰水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    weight = 1 + target_data.knowledge_interest[8]
    if 8 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[8])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.TARGET_FAITH_IS_LOW)
def handle_target_faith_is_low(character_id: int) -> int:
    """
    校验交互对象是否信仰水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 8 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[8])
        if level <= 2:
            return 1
        return 0
    return 1
