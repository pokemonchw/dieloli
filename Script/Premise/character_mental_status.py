from Script.Design import handle_premise, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.HAPPY_IS_HEIGHT)
def handle_happy_is_height(character_id: int) -> int:
    """
    校验角色是否快乐情绪高涨
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(8, 0)
    return character_data.status[8] > 100


@handle_premise.add_premise(constant.Premise.HAPPY_IS_LOW)
def handle_happy_is_low(character_id: int) -> int:
    """
    校验角色是否快乐情绪低下
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(8, 0)
    return character_data.status[8] < 10


@handle_premise.add_premise(constant.Premise.PAIN_IS_HEIGHT)
def handle_pain_is_height(character_id: int) -> int:
    """
    校验角色是否痛苦情绪高涨
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(9, 0)
    return character_data.status[9] > 100


@handle_premise.add_premise(constant.Premise.PAIN_IS_LOW)
def handle_pain_is_low(character_id: int) -> int:
    """
    校验角色是否痛苦情绪低下
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(9, 0)
    return character_data.status[9] < 10


@handle_premise.add_premise(constant.Premise.YEARN_IS_HEIGHT)
def handle_yearn_is_height(character_id: int) -> int:
    """
    校验角色是否渴望情绪高涨
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(10, 0)
    return character_data.status[10] > 100


@handle_premise.add_premise(constant.Premise.YEARN_IS_LOW)
def handle_yearn_is_low(character_id: int) -> int:
    """
    校验角色是否渴望情绪低下
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(10, 0)
    return character_data.status[10] < 10


@handle_premise.add_premise(constant.Premise.FEAR_IS_HEIGHT)
def handle_fear_is_height(character_id: int) -> int:
    """
    校验角色是否恐惧情绪高涨
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(11, 0)
    return character_data.status[11] > 100


@handle_premise.add_premise(constant.Premise.FEAR_IS_LOW)
def handle_fear_is_low(character_id: int) -> int:
    """
    校验角色是否恐惧情绪低下
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(11, 0)
    return character_data.status[11] < 10


@handle_premise.add_premise(constant.Premise.ANTIPATHY_IS_HEIGHT)
def handle_antipathy_is_height(character_id: int) -> int:
    """
    校验角色是否反感情绪高涨
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(12, 0)
    return character_data.status[12] > 100


@handle_premise.add_premise(constant.Premise.ANTIPATHY_IS_LOW)
def handle_antipathy_is_low(character_id: int) -> int:
    """
    校验角色是否反感情绪低下
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(12, 0)
    return character_data.status[12] < 10


@handle_premise.add_premise(constant.Premise.SHAME_IS_HEIGHT)
def handle_shame_is_height(character_id: int) -> int:
    """
    校验角色是否羞耻情绪高涨
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(13, 0)
    return character_data.status[13] > 100


@handle_premise.add_premise(constant.Premise.SHAME_IS_LOW)
def handle_shame_is_low(character_id: int) -> int:
    """
    校验角色是否羞耻情绪低下
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(13, 0)
    return character_data.status[13] < 10


@handle_premise.add_premise(constant.Premise.DEPRESSED_IS_HEIGHT)
def handle_depressed_is_height(character_id: int) -> int:
    """
    校验角色是否抑郁情绪高涨
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(14, 0)
    return character_data.status[14] > 100


@handle_premise.add_premise(constant.Premise.DEPRESSED_IS_LOW)
def handle_depressed_is_low(character_id: int) -> int:
    """
    校验角色是否抑郁情绪低下
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(14, 0)
    return character_data.status[14] < 10


@handle_premise.add_premise(constant.Premise.ARROGANT_IS_HEIGHT)
def handle_arrogant_is_height(character_id: int) -> int:
    """
    校验角色是否傲慢情绪高涨
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(15, 0)
    return character_data.status[15] > 100


@handle_premise.add_premise(constant.Premise.ARROGANT_IS_LOW)
def handle_arrogant_is_low(character_id: int) -> int:
    """
    校验角色是否傲慢情绪低下
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(15, 0)
    return character_data.status[15] < 10


@handle_premise.add_premise(constant.Premise.ENVY_IS_HEIGHT)
def handle_envy_is_height(character_id: int) -> int:
    """
    校验角色是否嫉妒情绪高涨
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(16, 0)
    return character_data.status[16] > 100


@handle_premise.add_premise(constant.Premise.ENVY_IS_LOW)
def handle_envy_is_low(character_id: int) -> int:
    """
    校验角色是否嫉妒情绪低下
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(16, 0)
    return character_data.status[16] < 10


@handle_premise.add_premise(constant.Premise.RAGE_IS_HEIGHT)
def handle_rage_is_height(character_id: int) -> int:
    """
    校验角色是否暴怒情绪高涨
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(17, 0)
    return character_data.status[17] > 100


@handle_premise.add_premise(constant.Premise.RAGE_IS_LOW)
def handle_rage_is_low(character_id: int) -> int:
    """
    校验角色是否暴怒情绪低下
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(17, 0)
    return character_data.status[17] < 10


@handle_premise.add_premise(constant.Premise.LAZY_IS_HEIGHT)
def handle_lazy_is_height(character_id: int) -> int:
    """
    校验角色是否懒惰情绪高涨
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(18, 0)
    return character_data.status[18] > 100


@handle_premise.add_premise(constant.Premise.LAZY_IS_LOW)
def handle_lazy_is_low(character_id: int) -> int:
    """
    校验角色是否懒惰情绪低下
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(18, 0)
    return character_data.status[18] < 10


@handle_premise.add_premise(constant.Premise.GREEDY_IS_HEIGHT)
def handle_greedy_is_height(character_id: int) -> int:
    """
    校验角色是否贪婪情绪高涨
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(19, 0)
    return character_data.status[19] > 100


@handle_premise.add_premise(constant.Premise.GREEDY_IS_LOW)
def handle_greedy_is_low(character_id: int) -> int:
    """
    校验角色是否贪婪情绪低下
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(19, 0)
    return character_data.status[19] < 10


@handle_premise.add_premise(constant.Premise.GLUTTONY_IS_HEIGHT)
def handle_gluttony_is_height(character_id: int) -> int:
    """
    校验角色是否暴食情绪高涨
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(20, 0)
    return character_data.status[20] > 100


@handle_premise.add_premise(constant.Premise.GLUTTONY_IS_LOW)
def handle_gluttony_is_low(character_id: int) -> int:
    """
    校验角色是否暴食情绪低下
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(20, 0)
    return character_data.status[20] < 10


@handle_premise.add_premise(constant.Premise.LUST_IS_HIGHT)
def handle_lust_is_hight(character_id: int) -> int:
    """
    校验角色是否色欲高涨
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(21, 0)
    return character_data.status[21] > 100


@handle_premise.add_premise(constant.Premise.LUST_IS_LOW)
def handle_lust_is_low(character_id: int) -> int:
    """
    校验角色是否色欲低下
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.status.setdefault(21, 0)
    return character_data.status[21] < 10
