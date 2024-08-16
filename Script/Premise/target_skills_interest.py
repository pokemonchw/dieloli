from Script.Design import handle_premise, attr_calculation, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.TARGET_SEXUAL_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_sexual_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否性技天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 9 in target_data.knowledge_interest:
        if target_data.knowledge_interest[9] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_SEXUAL_SKILLS_INTEREST_IS_LOW)
def handle_target_sexual_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否性技天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 9 in target_data.knowledge_interest:
        if target_data.knowledge_interest[9] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_COMPUTER_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_computer_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否计算机天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 10 in target_data.knowledge_interest:
        if target_data.knowledge_interest[10] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_COMPUTER_SKILLS_INTEREST_IS_LOW)
def handle_target_computer_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否计算机天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 10 in target_data.knowledge_interest:
        if target_data.knowledge_interest[10] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_PERFORMANCE_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_performance_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否表演天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 11 in target_data.knowledge_interest:
        if target_data.knowledge_interest[11] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_PERFORMANCE_SKILLS_INTEREST_IS_LOW)
def handle_target_performance_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否表演天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 11 in target_data.knowledge_interest:
        if target_data.knowledge_interest[11] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_ELOQUENCE_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_eloquence_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否口才天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 12 in target_data.knowledge_interest:
        if target_data.knowledge_interest[12] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_ELOQUENCE_SKILLS_INTEREST_IS_LOW)
def handle_target_eloquence_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否口才天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 12 in target_data.knowledge_interest:
        if target_data.knowledge_interest[12] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_PAINTING_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_painting_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否绘画天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 13 in target_data.knowledge_interest:
        if target_data.knowledge_interest[13] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_PAINTING_SKILLS_INTEREST_IS_LOW)
def handle_target_painting_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否绘画天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 13 in target_data.knowledge_interest:
        if target_data.knowledge_interest[13] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_SHOOT_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_ethic_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否拍摄天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 14 in target_data.knowledge_interest:
        if target_data.knowledge_interest[14] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_SHOOT_SKILLS_INTEREST_IS_LOW)
def handle_target_shoot_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否拍摄天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 14 in target_data.knowledge_interest:
        if target_data.knowledge_interest[14] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_SINGING_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_singing_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否演唱天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 15 in target_data.knowledge_interest:
        if target_data.knowledge_interest[15] >= 1:
            return 1
    return 0


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
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 15 in target_data.knowledge_interest:
        if target_data.knowledge_interest[15] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_WRITE_MUSIC_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_write_music_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否作曲天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 16 in target_data.knowledge_interest:
        if target_data.knowledge_interest[16] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_WRITE_MUSIC_SKILLS_INTEREST_IS_LOW)
def handle_target_write_music_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否作曲天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 16 in target_data.knowledge_interest:
        if target_data.knowledge_interest[16] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_COOKING_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_cooking_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否厨艺天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 17 in target_data.knowledge_interest:
        if target_data.knowledge_interest[17] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_COOKING_SKILLS_INTEREST_IS_LOW)
def handle_target_cooking_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否厨艺天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 17 in target_data.knowledge_interest:
        if target_data.knowledge_interest[17] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_DANCE_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_dance_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否舞蹈天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 18 in target_data.knowledge_interest:
        if target_data.knowledge_interest[18] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_DANCE_SKILLS_INTEREST_IS_LOW)
def handle_target_dance_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否舞蹈天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 18 in target_data.knowledge_interest:
        if target_data.knowledge_interest[18] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_TAILOR_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_tailor_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否裁缝天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 19 in target_data.knowledge_interest:
        if target_data.knowledge_interest[19] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_TAILOR_SKILLS_INTEREST_IS_LOW)
def handle_target_tailor_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否裁缝天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 19 in target_data.knowledge_interest:
        if target_data.knowledge_interest[19] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_TACTICS_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_tactics_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否战术天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 20 in target_data.knowledge_interest:
        if target_data.knowledge_interest[20] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_TACTICS_SKILLS_INTEREST_IS_LOW)
def handle_target_tactics_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否战术天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 20 in target_data.knowledge_interest:
        if target_data.knowledge_interest[20] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_SWIMMING_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_swimming_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否游泳天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 21 in target_data.knowledge_interest:
        if target_data.knowledge_interest[21] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_SWIMMING_SKILLS_INTEREST_IS_LOW)
def handle_target_swimming_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否游泳天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 21 in target_data.knowledge_interest:
        if target_data.knowledge_interest[21] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_MANUFACTURE_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_manufacture_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否制造天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 22 in target_data.knowledge_interest:
        if target_data.knowledge_interest[22] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_MANUFACTURE_SKILLS_INTEREST_IS_LOW)
def handle_target_manufacture_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否制造天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 22 in target_data.knowledge_interest:
        if target_data.knowledge_interest[22] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_FIRST_AID_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_first_aid_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否急救天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 23 in target_data.knowledge_interest:
        if target_data.knowledge_interest[23] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_FIRST_AID_SKILLS_INTEREST_IS_LOW)
def handle_target_first_aid_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否急救天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 23 in target_data.knowledge_interest:
        if target_data.knowledge_interest[23] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_ANATOMY_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_anatomy_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否解剖天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 24 in target_data.knowledge_interest:
        if target_data.knowledge_interest[24] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_ANATOMY_SKILLS_INTEREST_IS_LOW)
def handle_target_anatomy_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否解剖天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 24 in target_data.knowledge_interest:
        if target_data.knowledge_interest[24] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_PLAY_MUSIC_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_play_music_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否演奏天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 25 in target_data.knowledge_interest:
        if target_data.knowledge_interest[25] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_PLAY_MUSIC_SKILLS_INTEREST_IS_LOW)
def handle_target_play_music_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否演奏天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 25 in target_data.knowledge_interest:
        if target_data.knowledge_interest[25] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_PROGRAMMING_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_programming_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否编程天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 26 in target_data.knowledge_interest:
        if target_data.knowledge_interest[26] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_PROGRAMMING_SKILLS_INTEREST_IS_LOW)
def handle_target_programming_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否编程天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 26 in target_data.knowledge_interest:
        if target_data.knowledge_interest[26] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_HACKER_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_hacker_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否黑客天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 27 in target_data.knowledge_interest:
        if target_data.knowledge_interest[27] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_HACKER_SKILLS_INTEREST_IS_LOW)
def handle_target_hacker_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否黑客天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 27 in target_data.knowledge_interest:
        if target_data.knowledge_interest[27] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_WRITE_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_write_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否写作天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 28 in target_data.knowledge_interest:
        if target_data.knowledge_interest[28] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_WRITE_SKILLS_INTEREST_IS_LOW)
def handle_target_write_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否写作天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 28 in target_data.knowledge_interest:
        if target_data.knowledge_interest[28] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_TRANSACTION_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_transction_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否交易天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 29 in target_data.knowledge_interest:
        if target_data.knowledge_interest[29] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_TRANSACTION_SKILLS_INTEREST_IS_LOW)
def handle_target_transction_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否交易天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 29 in target_data.knowledge_interest:
        if target_data.knowledge_interest[29] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_CEREMONY_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_ceremony_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否礼仪天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 30 in target_data.knowledge_interest:
        if target_data.knowledge_interest[30] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_CEREMONY_SKILLS_INTEREST_IS_LOW)
def handle_target_ceremony_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否礼仪天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 30 in target_data.knowledge_interest:
        if target_data.knowledge_interest[30] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_MOTION_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_motion_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否运动天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 31 in target_data.knowledge_interest:
        if target_data.knowledge_interest[31] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_MOTION_SKILLS_INTEREST_IS_LOW)
def handle_target_motion_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否运动天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 31 in target_data.knowledge_interest:
        if target_data.knowledge_interest[31] < 1:
            return 1
        return 0
    return 1
