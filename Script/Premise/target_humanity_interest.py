from Script.Design import handle_premise, attr_calculation, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.TARGET_ETHIC_INTEREST_IS_HEIGHT)
def handle_target_ethic_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否伦理天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 0 in target_data.knowledge_interest:
        if target_data.knowledge_interest[0] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_ETHIC_INTEREST_IS_LOW)
def handle_target_ethic_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否伦理天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 0 in target_data.knowledge_interest:
        if target_data.knowledge_interest[0] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_MORALITY_INTEREST_IS_HEIGHT)
def handle_target_morality_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否道德天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 1 in target_data.knowledge_interest:
        if target_data.knowledge_interest[1] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_MORALITY_INTEREST_IS_LOW)
def handle_target_morality_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否道德天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 1 in target_data.knowledge_interest:
        if target_data.knowledge_interest[1] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_LITERATURE_INTEREST_IS_HEIGHT)
def handle_target_literature_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否文学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 2 in target_data.knowledge_interest:
        if target_data.knowledge_interest[2] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_LITERATURE_INTEREST_IS_LOW)
def handle_target_literature_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否文学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 2 in target_data.knowledge_interest:
        if target_data.knowledge_interest[2] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_POETRY_INTEREST_IS_HEIGHT)
def handle_target_poetry_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否诗歌天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 3 in target_data.knowledge_interest:
        if target_data.knowledge_interest[3] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_POETRY_INTEREST_IS_LOW)
def handle_target_poetry_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否诗歌天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 3 in target_data.knowledge_interest:
        if target_data.knowledge_interest[3] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_HISTORY_INTEREST_IS_HEIGHT)
def handle_target_history_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否历史天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 4 in target_data.knowledge_interest:
        if target_data.knowledge_interest[4] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_HISTORY_INTEREST_IS_LOW)
def handle_target_history_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否历史天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 4 in target_data.knowledge_interest:
        if target_data.knowledge_interest[4] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_ART_INTEREST_IS_HEIGHT)
def handle_target_art_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否艺术天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 5 in target_data.knowledge_interest:
        if target_data.knowledge_interest[5] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_ART_INTEREST_IS_LOW)
def handle_target_art_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否艺术天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 5 in target_data.knowledge_interest:
        if target_data.knowledge_interest[5] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_MUSIC_THEORY_INTEREST_IS_HEIGHT)
def handle_target_music_theory_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否乐理天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 6 in target_data.knowledge_interest:
        if target_data.knowledge_interest[6] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_MUSIC_THEORY_INTEREST_IS_LOW)
def handle_target_music_theory_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否乐理天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 6 in target_data.knowledge_interest:
        if target_data.knowledge_interest[6] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_RELIGION_INTEREST_IS_HEIGHT)
def handle_target_religion_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否宗教天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 7 in target_data.knowledge_interest:
        if target_data.knowledge_interest[7] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_RELIGION_INTEREST_IS_LOW)
def handle_target_religion_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否宗教天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 7 in target_data.knowledge_interest:
        if target_data.knowledge_interest[7] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_FAITH_INTEREST_IS_HEIGHT)
def handle_target_faith_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否信仰天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 8 in target_data.knowledge_interest:
        if target_data.knowledge[8] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_FAITH_INTEREST_IS_LOW)
def handle_target_faith_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否信仰天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 8 in target_data.knowledge_interest:
        if target_data.knowledge_interest[8] < 1:
            return 1
    return 1
