
from Script.Design import handle_premise, attr_calculation, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.CHINESE_SKILLS_INTEREST_IS_HEIGHT)
def handle_chinese_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否汉语天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 0 in character_data.language_interest:
        if character_data.language_interest[0] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.CHINESE_SKILLS_INTEREST_IS_LOW)
def handle_chinese_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否汉语天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 0 in character_data.language_interest:
        if character_data.language_interest[0] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.ENGLISH_SKILLS_INTEREST_IS_HEIGHT)
def handle_english_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否英语天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 1 in character_data.language_interest:
        if character_data.language_interest[1] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.ENGLISH_SKILLS_INTEREST_IS_LOW)
def handle_english_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否英语天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 1 in character_data.language_interest:
        if character_data.language_interest[1] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.FRENCH_SKILLS_INTEREST_IS_HEIGHT)
def handle_french_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否法语天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 2 in character_data.language_interest:
        if character_data.language_interest[2] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.FRENCH_SKILLS_INTEREST_IS_LOW)
def handle_french_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否法语天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 2 in character_data.language_interest:
        if character_data.language_interest[2] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.SPANISH_SKILLS_INTEREST_IS_HEIGHT)
def handle_spanish_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否西班牙语天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 3 in character_data.language_interest:
        if character_data.language_interest[3] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.SPANISH_SKILLS_INTEREST_IS_LOW)
def handle_spanish_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否西班牙语天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 3 in character_data.language_interest:
        if character_data.language_interest[3] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.ARABIC_SKILLS_INTEREST_IS_HEIGHT)
def handle_arabic_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否阿拉伯语天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 4 in character_data.language_interest:
        if character_data.language_interest[4] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.ARABIC_SKILLS_INTEREST_IS_LOW)
def handle_arabic_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否阿拉伯语天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 4 in character_data.language_interest:
        if character_data.language_interest[4] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.RUSSIAN_SKILLS_INTEREST_IS_HEIGHT)
def handle_russian_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否俄语天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 5 in character_data.language_interest:
        if character_data.language_interest[5] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.RUSSIAN_SKILLS_INTEREST_IS_LOW)
def handle_russian_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否俄语天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 5 in character_data.language_interest:
        if character_data.language_interest[5] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.JAPANESE_SKILLS_INTEREST_IS_HEIGHT)
def handle_japanese_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否日语天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 6 in character_data.language_interest:
        if character_data.language_interest[6] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.JAPANESE_SKILLS_INTEREST_IS_LOW)
def handle_japanese_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否日语天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 6 in character_data.language_interest:
        if character_data.language_interest[6] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.HINDI_SKILLS_INTEREST_IS_HEIGHT)
def handle_hindi_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否印地语天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 7 in character_data.language_interest:
        if character_data.language_interest[7] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.HINDI_SKILLS_INTEREST_IS_LOW)
def handle_hindi_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否印地语天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 7 in character_data.language_interest:
        if character_data.language_interest[7] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.OLD_LATIN_SKILLS_INTEREST_IS_HEIGHT)
def handle_old_latin_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否古拉丁语天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 8 in character_data.language_interest:
        if character_data.language_interest[8] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.OLD_LATIN_SKILLS_INTEREST_IS_LOW)
def handle_old_latin_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否古拉丁语天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 8 in character_data.language_interest:
        if character_data.language_interest[8] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.LATIN_SKILLS_INTEREST_IS_HEIGHT)
def handle_latin_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否拉丁语天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 9 in character_data.language_interest:
        if character_data.language_interest[9] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.LATIN_SKILLS_INTEREST_IS_LOW)
def handle_latin_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否拉丁语天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 9 in character_data.language_interest:
        if character_data.language_interest[9] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.ANCIENT_CHINESE_SKILLS_INTEREST_IS_HEIGHT)
def handle_ancient_chinese_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否古汉语天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 10 in character_data.language_interest:
        if character_data.language_interest[10] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.ANCIENT_CHINESE_SKILLS_INTEREST_IS_LOW)
def handle_ancient_chinese_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否古汉语天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 10 in character_data.language_interest:
        if character_data.language_interest[10] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.OLD_SINITIC_SKILLS_INTEREST_IS_HEIGHT)
def handle_old_sinitic_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否上古汉语天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 11 in character_data.language_interest:
        if character_data.language_interest[11] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.OLD_SINITIC_SKILLS_INTEREST_IS_LOW)
def handle_old_sinitic_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否上古汉语天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 11 in character_data.language_interest:
        if character_data.language_interest[11] < 1:
            return 1
    return 1
