from Script.Design import handle_premise, attr_calculation, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.SEXUAL_SKILLS_IS_HEIGHT)
def handle_sexual_skills_is_height(character_id: int) -> int:
    """
    校验角色是否性技水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 9 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[9])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.SEXUAL_SKILLS_IS_LOW)
def handle_sexual_skills_is_low(character_id: int) -> int:
    """
    校验角色是否性技水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 9 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[9])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.COMPUTER_SKILLS_IS_HEIGHT)
def handle_computer_skills_is_height(character_id: int) -> int:
    """
    校验角色是否计算机水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 10 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[10])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.COMPUTER_SKILLS_IS_LOW)
def handle_computer_skills_is_low(character_id: int) -> int:
    """
    校验角色是否计算机水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 10 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[10])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.PERFORMANCE_SKILLS_IS_HEIGHT)
def handle_performance_is_height(character_id: int) -> int:
    """
    校验角色是否表演水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 11 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[11])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.PERFORMANCE_SKILLS_IS_LOW)
def handle_performance_is_low(character_id: int) -> int:
    """
    校验角色是否表演水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 11 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[11])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.ELOQUENCE_SKILLS_IS_HEIGHT)
def handle_eloquence_skills_is_height(character_id: int) -> int:
    """
    校验角色是否口才水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 12 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[12])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.ELOQUENCE_SKILLS_IS_LOW)
def handle_eloquence_is_low(character_id: int) -> int:
    """
    校验角色是否口才水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 12 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[12])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.PAINTING_SKILLS_IS_HEIGHT)
def handle_painting_skills_is_height(character_id: int) -> int:
    """
    校验角色是否绘画水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 13 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[13])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.PAINTING_SKILLS_IS_LOW)
def handle_painting_skills_is_low(character_id: int) -> int:
    """
    校验角色是否绘画水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 13 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[13])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.SHOOT_SKILLS_IS_HEIGHT)
def handle_ethic_is_height(character_id: int) -> int:
    """
    校验角色是否拍摄水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 14 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[14])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.SHOOT_SKILLS_IS_LOW)
def handle_shoot_skills_is_low(character_id: int) -> int:
    """
    校验角色是否拍摄水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 14 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[14])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.SINGING_SKILLS_IS_HEIGHT)
def handle_singing_skills_is_height(character_id: int) -> int:
    """
    校验角色是否演唱水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 15 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[15])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.SINGING_SKILLS_IS_LOW)
def handle_singing_is_low(character_id: int) -> int:
    """
    校验角色是否演唱水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 15 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[15])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.WRITE_MUSIC_SKILLS_IS_HEIGHT)
def handle_write_music_skills_is_height(character_id: int) -> int:
    """
    校验角色是否作曲水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 16 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[16])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.WRITE_MUSIC_SKILLS_IS_LOW)
def handle_write_music_skills_is_low(character_id: int) -> int:
    """
    校验角色是否作曲水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 16 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[16])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.COOKING_SKILLS_IS_HEIGHT)
def handle_cooking_skills_is_height(character_id: int) -> int:
    """
    校验角色是否厨艺水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 17 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[17])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.COOKING_SKILLS_IS_LOW)
def handle_cooking_skills_is_low(character_id: int) -> int:
    """
    校验角色是否厨艺水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 17 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[17])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.DANCE_SKILLS_IS_HEIGHT)
def handle_dance_skills_is_height(character_id: int) -> int:
    """
    校验角色是否舞蹈水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 18 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[18])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.DANCE_SKILLS_IS_LOW)
def handle_dance_skills_is_low(character_id: int) -> int:
    """
    校验角色是否舞蹈水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 18 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[18])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TAILOR_SKILLS_IS_HEIGHT)
def handle_tailor_skills_is_height(character_id: int) -> int:
    """
    校验角色是否裁缝水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 19 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[19])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TAILOR_SKILLS_IS_LOW)
def handle_tailor_skills_is_low(character_id: int) -> int:
    """
    校验角色是否裁缝水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 19 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[19])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TACTICS_SKILLS_IS_HEIGHT)
def handle_tactics_skills_is_height(character_id: int) -> int:
    """
    校验角色是否战术水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 20 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[20])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TACTICS_SKILLS_IS_LOW)
def handle_tactics_is_low(character_id: int) -> int:
    """
    校验角色是否战术水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 20 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[20])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.SWIMMING_SKILLS_IS_HEIGHT)
def handle_swimming_skills_is_height(character_id: int) -> int:
    """
    校验角色是否游泳水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 21 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[21])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.SWIMMING_SKILLS_IS_LOW)
def handle_swimming_skills_is_low(character_id: int) -> int:
    """
    校验角色是否游泳水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 21 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[21])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.MANUFACTURE_SKILLS_IS_HEIGHT)
def handle_manufacture_skills_is_height(character_id: int) -> int:
    """
    校验角色是否制造水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 22 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[22])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.MANUFACTURE_SKILLS_IS_LOW)
def handle_manufacture_skills_is_low(character_id: int) -> int:
    """
    校验角色是否制造水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 22 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[22])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.FIRST_AID_SKILLS_IS_HEIGHT)
def handle_first_aid_skills_is_height(character_id: int) -> int:
    """
    校验角色是否急救水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 23 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[23])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.FIRST_AID_SKILLS_IS_LOW)
def handle_first_aid_skills_is_low(character_id: int) -> int:
    """
    校验角色是否急救水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 23 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[23])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.ANATOMY_SKILLS_IS_HEIGHT)
def handle_anatomy_skills_is_height(character_id: int) -> int:
    """
    校验角色是否解剖水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 24 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[24])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.ANATOMY_SKILLS_IS_LOW)
def handle_anatomy_skills_is_low(character_id: int) -> int:
    """
    校验角色是否解剖水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 24 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[24])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.PLAY_MUSIC_SKILLS_IS_HEIGHT)
def handle_play_music_skills_is_height(character_id: int) -> int:
    """
    校验角色是否演奏水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 25 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[25])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.PLAY_MUSIC_SKILLS_IS_LOW)
def handle_play_music_skills_is_low(character_id: int) -> int:
    """
    校验角色是否演奏水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 25 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[25])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.PROGRAMMING_SKILLS_IS_HEIGHT)
def handle_programming_skills_is_height(character_id: int) -> int:
    """
    校验角色是否编程水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 26 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[26])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.PROGRAMMING_SKILLS_IS_LOW)
def handle_programming_skills_is_low(character_id: int) -> int:
    """
    校验角色是否编程水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 26 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[26])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.HACKER_SKILLS_IS_HEIGHT)
def handle_hacker_skills_is_height(character_id: int) -> int:
    """
    校验角色是否黑客水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 27 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[27])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.HACKER_SKILLS_IS_LOW)
def handle_hacker_skills_is_low(character_id: int) -> int:
    """
    校验角色是否黑客水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 27 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[27])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.WRITE_SKILLS_IS_HEIGHT)
def handle_write_skills_is_height(character_id: int) -> int:
    """
    校验角色是否写作水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 28 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[28])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.WRITE_SKILLS_IS_LOW)
def handle_write_skills_is_low(character_id: int) -> int:
    """
    校验角色是否写作水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 28 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[28])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TRANSACTION_SKILLS_IS_HEIGHT)
def handle_transction_skills_is_height(character_id: int) -> int:
    """
    校验角色是否交易水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 29 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[29])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TRANSACTION_SKILLS_IS_LOW)
def handle_transction_skills_is_low(character_id: int) -> int:
    """
    校验角色是否交易水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 29 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[29])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.CEREMONY_SKILLS_IS_HEIGHT)
def handle_ceremony_skills_is_height(character_id: int) -> int:
    """
    校验角色是否礼仪水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 30 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[30])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.CEREMONY_SKILLS_IS_LOW)
def handle_ceremony_skills_is_low(character_id: int) -> int:
    """
    校验角色是否礼仪水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 30 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[30])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.MOTION_SKILLS_IS_HEIGHT)
def handle_motion_skills_is_height(character_id: int) -> int:
    """
    校验角色是否运动水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 31 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[31])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.MOTION_SKILLS_IS_LOW)
def handle_motion_skills_is_low(character_id: int) -> int:
    """
    校验角色是否运动水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 31 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[31])
        return level < 3
    return 1
