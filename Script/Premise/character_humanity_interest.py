from Script.Design import handle_premise, attr_calculation, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.ETHIC_INTEREST_IS_HEIGHT)
def handle_ethic_interest_is_height(character_id: int) -> int:
    """
    校验角色是否伦理天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 0 in character_data.knowledge_interest:
        if character_data.knowledge_interest[0] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.ETHIC_INTEREST_IS_LOW)
def handle_ethic_interest_is_low(character_id: int) -> int:
    """
    校验角色是否伦理天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 0 in character_data.knowledge_interest:
        if character_data.knowledge_interest[0] < 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.MORALITY_INTEREST_IS_HEIGHT)
def handle_morality_interest_is_height(character_id: int) -> int:
    """
    校验角色是否道德天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[1]
    if 1 in character_data.knowledge_interest:
        if character_data.knowledge_interest[1] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.MORALITY_INTEREST_IS_LOW)
def handle_morality_interest_is_low(character_id: int) -> int:
    """
    校验角色是否道德天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 1 in character_data.knowledge_interest:
        if character_data.knowledge_interest[1] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.LITERATURE_INTEREST_IS_HEIGHT)
def handle_literature_interest_is_height(character_id: int) -> int:
    """
    校验角色是否文学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 2 in character_data.knowledge_interest:
        if character_data.knowledge_interest[2] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.LITERATURE_INTEREST_IS_LOW)
def handle_literature_interest_is_low(character_id: int) -> int:
    """
    校验角色是否文学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 2 in character_data.knowledge_interest:
        if character_data.knowledge_interest[2] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.POETRY_INTEREST_IS_HEIGHT)
def handle_poetry_interest_is_height(character_id: int) -> int:
    """
    校验角色是否诗歌天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 3 in character_data.knowledge_interest:
        if character_data.knowledge_interest[3] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.POETRY_INTEREST_IS_LOW)
def handle_poetry_interest_is_low(character_id: int) -> int:
    """
    校验角色是否诗歌天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 3 in character_data.knowledge_interest:
        if character_data.knowledge_interest[3] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.HISTORY_INTEREST_IS_HEIGHT)
def handle_history_interest_is_height(character_id: int) -> int:
    """
    校验角色是否历史天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 4 in character_data.knowledge_interest:
        if character_data.knowledge_interest[4] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.HISTORY_INTEREST_IS_LOW)
def handle_history_interest_is_low(character_id: int) -> int:
    """
    校验角色是否历史天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 4 in character_data.knowledge_interest:
        if character_data.knowledge_interest[4] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.ART_INTEREST_IS_HEIGHT)
def handle_art_interest_is_height(character_id: int) -> int:
    """
    校验角色是否艺术天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 5 in character_data.knowledge_interest:
        if character_data.knowledge_interest[5] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.ART_INTEREST_IS_LOW)
def handle_art_interest_is_low(character_id: int) -> int:
    """
    校验角色是否艺术天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 5 in character_data.knowledge_interest:
        if character_data.knowledge_interest[5] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.MUSIC_THEORY_INTEREST_IS_HEIGHT)
def handle_music_theory_interest_is_height(character_id: int) -> int:
    """
    校验角色是否乐理天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 6 in character_data.knowledge_interest:
        if character_data.knowledge_interest[6] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.MUSIC_THEORY_INTEREST_IS_LOW)
def handle_music_theory_interest_is_low(character_id: int) -> int:
    """
    校验角色是否乐理天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 6 in character_data.knowledge_interest:
        if character_data.knowledge_interest[6] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.RELIGION_INTEREST_IS_HEIGHT)
def handle_religion_interest_is_height(character_id: int) -> int:
    """
    校验角色是否宗教天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 7 in character_data.knowledge_interest:
        if character_data.knowledge_interest[7] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.RELIGION_INTEREST_IS_LOW)
def handle_religion_interest_is_low(character_id: int) -> int:
    """
    校验角色是否宗教天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 7 in character_data.knowledge_interest:
        if character_data.knowledge_interest[7] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.FAITH_INTEREST_IS_HEIGHT)
def handle_faith_interest_is_height(character_id: int) -> int:
    """
    校验角色是否信仰天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 8 in character_data.knowledge_interest:
        if character_data.knowledge_interest[8] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.FAITH_INTEREST_IS_LOW)
def handle_faith_interest_is_low(character_id: int) -> int:
    """
    校验角色是否信仰天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 8 in character_data.knowledge_interest:
        if character_data.knowledge_interest[8] < 1:
            return 1
    return 1
