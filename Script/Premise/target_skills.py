from Script.Design import handle_premise, attr_calculation, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.TARGET_SEXUAL_SKILLS_IS_HEIGHT)
def handle_target_sexual_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否性技水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 9 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[9])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_SEXUAL_SKILLS_IS_LOW)
def handle_target_sexual_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否性技水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 9 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[9])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_COMPUTER_SKILLS_IS_HEIGHT)
def handle_target_computer_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否计算机水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 10 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[10])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_COMPUTER_SKILLS_IS_LOW)
def handle_target_computer_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否计算机水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 10 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[10])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_PERFORMANCE_SKILLS_IS_HEIGHT)
def handle_target_performance_is_height(character_id: int) -> int:
    """
    校验交互对象是否表演水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 11 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[11])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_PERFORMANCE_SKILLS_IS_LOW)
def handle_target_performance_is_low(character_id: int) -> int:
    """
    校验交互对象是否表演水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 11 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[11])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_ELOQUENCE_SKILLS_IS_HEIGHT)
def handle_target_eloquence_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否口才水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 12 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[12])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_ELOQUENCE_SKILLS_IS_LOW)
def handle_target_eloquence_is_low(character_id: int) -> int:
    """
    校验交互对象是否口才水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 12 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[12])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_PAINTING_SKILLS_IS_HEIGHT)
def handle_target_painting_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否绘画水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 13 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[13])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_PAINTING_SKILLS_IS_LOW)
def handle_target_painting_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否绘画水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 13 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[13])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_SHOOT_SKILLS_IS_HEIGHT)
def handle_target_ethic_is_height(character_id: int) -> int:
    """
    校验交互对象是否拍摄水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 14 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[14])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_SHOOT_SKILLS_IS_LOW)
def handle_target_shoot_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否拍摄水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 14 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[14])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_SINGING_SKILLS_IS_HEIGHT)
def handle_target_singing_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否演唱水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    weight = 1 + target_data.knowledge_interest[15]
    if 15 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[15])
        return level > 5
    return weight


@handle_premise.add_premise(constant.Premise.TARGET_SINGING_SKILLS_IS_LOW)
def handle_target_singing_is_low(character_id: int) -> int:
    """
    校验交互对象是否演唱水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 15 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[15])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_WRITE_MUSIC_SKILLS_IS_HEIGHT)
def handle_target_write_music_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否作曲水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 16 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[16])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_WRITE_MUSIC_SKILLS_IS_LOW)
def handle_target_write_music_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否作曲水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 16 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[16])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_COOKING_SKILLS_IS_HEIGHT)
def handle_target_cooking_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否厨艺水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 17 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[17])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_COOKING_SKILLS_IS_LOW)
def handle_target_cooking_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否厨艺水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 17 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[17])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_DANCE_SKILLS_IS_HEIGHT)
def handle_target_dance_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否舞蹈水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 18 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[18])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_DANCE_SKILLS_IS_LOW)
def handle_target_dance_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否舞蹈水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 18 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[18])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_TAILOR_SKILLS_IS_HEIGHT)
def handle_target_tailor_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否裁缝水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 19 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[19])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_TAILOR_SKILLS_IS_LOW)
def handle_target_tailor_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否裁缝水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 19 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[19])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_TACTICS_SKILLS_IS_HEIGHT)
def handle_target_tactics_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否战术水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 20 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[20])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_TACTICS_SKILLS_IS_LOW)
def handle_target_tactics_is_low(character_id: int) -> int:
    """
    校验交互对象是否战术水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 20 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[20])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_SWIMMING_SKILLS_IS_HEIGHT)
def handle_target_swimming_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否游泳水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 21 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[21])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_SWIMMING_SKILLS_IS_LOW)
def handle_target_swimming_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否游泳水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 21 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[21])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_MANUFACTURE_SKILLS_IS_HEIGHT)
def handle_target_manufacture_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否制造水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 22 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[22])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_MANUFACTURE_SKILLS_IS_LOW)
def handle_target_manufacture_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否制造水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 22 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[22])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_FIRST_AID_SKILLS_IS_HEIGHT)
def handle_target_first_aid_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否急救水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 23 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[23])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_FIRST_AID_SKILLS_IS_LOW)
def handle_target_first_aid_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否急救水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 23 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[23])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_ANATOMY_SKILLS_IS_HEIGHT)
def handle_target_anatomy_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否解剖水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 24 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[24])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_ANATOMY_SKILLS_IS_LOW)
def handle_target_anatomy_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否解剖水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 24 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[24])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_PLAY_MUSIC_SKILLS_IS_HEIGHT)
def handle_target_play_music_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否演奏水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 25 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[25])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_PLAY_MUSIC_SKILLS_IS_LOW)
def handle_target_play_music_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否演奏水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 25 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[25])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_PROGRAMMING_SKILLS_IS_HEIGHT)
def handle_target_programming_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否编程水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 26 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[26])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_PROGRAMMING_SKILLS_IS_LOW)
def handle_target_programming_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否编程水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 26 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[26])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_HACKER_SKILLS_IS_HEIGHT)
def handle_target_hacker_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否黑客水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 27 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[27])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_HACKER_SKILLS_IS_LOW)
def handle_target_hacker_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否黑客水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 27 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[27])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_WRITE_SKILLS_IS_HEIGHT)
def handle_target_write_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否写作水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 28 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[28])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_WRITE_SKILLS_IS_LOW)
def handle_target_write_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否写作水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 28 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[28])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_TRANSACTION_SKILLS_IS_HEIGHT)
def handle_target_transction_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否交易水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 29 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[29])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_TRANSACTION_SKILLS_IS_LOW)
def handle_target_transction_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否交易水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 29 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[29])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_CEREMONY_SKILLS_IS_HEIGHT)
def handle_target_ceremony_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否礼仪水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 30 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[30])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_CEREMONY_SKILLS_IS_LOW)
def handle_target_ceremony_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否礼仪水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 30 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[30])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_MOTION_SKILLS_IS_HEIGHT)
def handle_target_motion_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否运动水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 31 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[31])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_MOTION_SKILLS_IS_LOW)
def handle_target_motion_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否运动水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 31 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[31])
        return level < 3
    return 1
