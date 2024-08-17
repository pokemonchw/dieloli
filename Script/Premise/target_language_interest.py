from Script.Design import handle_premise, attr_calculation, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.TARGET_CHINESE_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_chinese_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否汉语天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 0 in target_data.language_interest:
        if target_data.language_interest[0] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_CHINESE_SKILLS_INTEREST_IS_LOW)
def handle_target_chinese_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否汉语天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 0 in target_data.language_interest:
        if target_data.language_interest[0] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_ENGLISH_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_english_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否英语天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 1 in target_data.language_interest:
        if target_data.language_interest[1] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_ENGLISH_SKILLS_INTEREST_IS_LOW)
def handle_target_english_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否英语天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 1 in target_data.language_interest:
        if target_data.language_interest[1] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_FRENCH_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_french_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否法语天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 2 in target_data.language_interest:
        if target_data.language_interest[2] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_FRENCH_SKILLS_INTEREST_IS_LOW)
def handle_target_french_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否法语天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 2 in target_data.language_interest:
        if target_data.language_interest[2] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_SPANISH_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_spanish_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否西班牙语天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 3 in target_data.language_interest:
        if target_data.language_interest[3] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_SPANISH_SKILLS_INTEREST_IS_LOW)
def handle_target_spanish_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否西班牙语天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 3 in target_data.language_interest:
        if target_data.language_interest[3] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_ARABIC_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_arabic_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否阿拉伯语天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 4 in target_data.language_interest:
        if target_data.language_interest[4] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_ARABIC_SKILLS_INTEREST_IS_LOW)
def handle_target_arabic_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否阿拉伯语天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 4 in target_data.language_interest:
        if target_data.language_interest[4] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_RUSSIAN_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_russian_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否俄语天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 5 in target_data.language_interest:
        if target_data.language_interest[5] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_RUSSIAN_SKILLS_INTEREST_IS_LOW)
def handle_target_russian_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否俄语天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 5 in target_data.language_interest:
        if target_data.language_interest[5] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_JAPANESE_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_japanese_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否日语天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 6 in target_data.language_interest:
        if target_data.language_interest[6] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_JAPANESE_SKILLS_INTEREST_IS_LOW)
def handle_target_japanese_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否日语天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 6 in target_data.language_interest:
        if target_data.language_interest[6] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_HINDI_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_hindi_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否印地语天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 7 in target_data.language_interest:
        if target_data.language_interest[7] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_HINDI_SKILLS_INTEREST_IS_LOW)
def handle_target_hindi_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否印地语天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 7 in target_data.language_interest:
        if target_data.language_interest[7] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_OLD_LATIN_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_old_latin_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否古拉丁语天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 8 in target_data.language_interest:
        if target_data.language_interest[8] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_OLD_LATIN_SKILLS_INTEREST_IS_LOW)
def handle_target_old_latin_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否古拉丁语天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 8 in target_data.language_interest:
        if target_data.language_interest[8] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_LATIN_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_latin_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否拉丁语天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 9 in target_data.language_interest:
        if target_data.language_interest[9] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_LATIN_SKILLS_INTEREST_IS_LOW)
def handle_target_latin_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否拉丁语天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 9 in target_data.language_interest:
        if target_data.language_interest[9] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_ANCIENT_CHINESE_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_ancient_chinese_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否古汉语天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 10 in target_data.language_interest:
        if target_data.language_interest[10] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_ANCIENT_CHINESE_SKILLS_INTEREST_IS_LOW)
def handle_target_ancient_chinese_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否古汉语天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 10 in target_data.language_interest:
        if target_data.language_interest[10] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_OLD_SINITIC_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_old_sinitic_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否上古汉语天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 11 in target_data.language_interest:
        if target_data.language_interest[11] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_OLD_SINITIC_SKILLS_INTEREST_IS_LOW)
def handle_target_old_sinitic_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否上古汉语天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 11 in target_data.language_interest:
        if target_data.language_interest[11] < 1:
            return 1
        return 0
    return 1
