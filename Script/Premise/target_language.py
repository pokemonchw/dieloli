from Script.Design import handle_premise, attr_calculation, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.TARGET_CHINESE_SKILLS_IS_HEIGHT)
def handle_target_chinese_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否汉语水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 0 in target_data.language:
        level = attr_calculation.get_experience_level_weight(target_data.language[0])
        if level > 5:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_CHINESE_SKILLS_IS_LOW)
def handle_target_chinese_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否汉语水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 0 in target_data.language:
        level = attr_calculation.get_experience_level_weight(target_data.language[0])
        if level < 3:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_ENGLISH_SKILLS_IS_HEIGHT)
def handle_target_english_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否英语水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 1 in target_data.language:
        level = attr_calculation.get_experience_level_weight(target_data.language[1])
        if level > 5:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_ENGLISH_SKILLS_IS_LOW)
def handle_target_english_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否英语水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 1 in target_data.language:
        level = attr_calculation.get_experience_level_weight(target_data.language[1])
        if level < 3:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_FRENCH_SKILLS_IS_HEIGHT)
def handle_target_french_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否法语水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 2 in target_data.language:
        level = attr_calculation.get_experience_level_weight(target_data.language[2])
        if level > 5:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_FRENCH_SKILLS_IS_LOW)
def handle_target_french_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否法语水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 2 in target_data.language:
        level = attr_calculation.get_experience_level_weight(target_data.language[2])
        if level < 3:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_SPANISH_SKILLS_IS_HEIGHT)
def handle_target_spanish_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否西班牙语水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 3 in target_data.language:
        level = attr_calculation.get_experience_level_weight(target_data.language[3])
        if level > 5:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_SPANISH_SKILLS_IS_LOW)
def handle_target_spanish_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否西班牙语水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 3 in target_data.language:
        level = attr_calculation.get_experience_level_weight(target_data.language[3])
        if level < 3:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_ARABIC_SKILLS_IS_HEIGHT)
def handle_target_arabic_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否阿拉伯语水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 4 in target_data.language:
        level = attr_calculation.get_experience_level_weight(target_data.language[4])
        if level > 5:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_ARABIC_SKILLS_IS_LOW)
def handle_target_arabic_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否阿拉伯语水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 4 in target_data.language:
        level = attr_calculation.get_experience_level_weight(target_data.language[4])
        if level < 3:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_RUSSIAN_SKILLS_IS_HEIGHT)
def handle_target_russian_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否俄语水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 5 in target_data.language:
        level = attr_calculation.get_experience_level_weight(target_data.language[5])
        if level > 5:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_RUSSIAN_SKILLS_IS_LOW)
def handle_target_russian_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否俄语水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 5 in target_data.language:
        level = attr_calculation.get_experience_level_weight(target_data.language[5])
        if level < 3:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_JAPANESE_SKILLS_IS_HEIGHT)
def handle_target_japanese_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否日语水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 6 in target_data.language:
        level = attr_calculation.get_experience_level_weight(target_data.language[6])
        if level > 5:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_JAPANESE_SKILLS_IS_LOW)
def handle_target_japanese_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否日语水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 6 in target_data.language:
        level = attr_calculation.get_experience_level_weight(target_data.language[6])
        if level < 3:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_HINDI_SKILLS_IS_HEIGHT)
def handle_target_hindi_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否印地语水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 7 in target_data.language:
        level = attr_calculation.get_experience_level_weight(target_data.language[7])
        if level > 5:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_HINDI_SKILLS_IS_LOW)
def handle_target_hindi_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否印地语水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 7 in target_data.language:
        level = attr_calculation.get_experience_level_weight(target_data.language[7])
        if level < 3:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_OLD_LATIN_SKILLS_IS_HEIGHT)
def handle_target_old_latin_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否古拉丁语水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 8 in target_data.language:
        level = attr_calculation.get_experience_level_weight(target_data.language[8])
        if level > 5:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_OLD_LATIN_SKILLS_IS_LOW)
def handle_target_old_latin_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否古拉丁语水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 8 in target_data.language:
        level = attr_calculation.get_experience_level_weight(target_data.language[8])
        if level < 3:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_LATIN_SKILLS_IS_HEIGHT)
def handle_target_latin_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否拉丁语水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 9 in target_data.language:
        level = attr_calculation.get_experience_level_weight(target_data.language[9])
        if level > 5:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_LATIN_SKILLS_IS_LOW)
def handle_target_latin_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否拉丁语水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 9 in target_data.language:
        level = attr_calculation.get_experience_level_weight(target_data.language[9])
        if level < 3:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_ANCIENT_CHINESE_SKILLS_IS_HEIGHT)
def handle_target_ancient_chinese_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否古汉语水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 10 in target_data.language:
        level = attr_calculation.get_experience_level_weight(target_data.language[10])
        if level > 5:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_ANCIENT_CHINESE_SKILLS_IS_LOW)
def handle_target_ancient_chinese_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否古汉语水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 10 in target_data.language:
        level = attr_calculation.get_experience_level_weight(target_data.language[10])
        if level < 3:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_OLD_SINITIC_SKILLS_IS_HEIGHT)
def handle_target_old_sinitic_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否上古汉语水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 11 in target_data.language:
        level = attr_calculation.get_experience_level_weight(target_data.language[11])
        if level > 5:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_OLD_SINITIC_SKILLS_IS_LOW)
def handle_target_old_sinitic_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否上古汉语水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 11 in target_data.language:
        level = attr_calculation.get_experience_level_weight(target_data.language[11])
        if level < 3:
            return 1
        return 0
    return 1
