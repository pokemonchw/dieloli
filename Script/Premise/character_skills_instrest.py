from Script.Design import handle_premise, attr_calculation, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.SEXUAL_SKILLS_INTEREST_IS_HEIGHT)
def handle_sexual_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否性技天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 9 in character_data.knowledge_interest:
        if character_data.knowledge_interest[9] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.SEXUAL_SKILLS_INTEREST_IS_LOW)
def handle_sexual_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否性技天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 9 in character_data.knowledge_interest:
        if character_data.knowledge_interest[9] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.COMPUTER_SKILLS_INTEREST_IS_HEIGHT)
def handle_computer_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否计算机天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 10 in character_data.knowledge_interest:
        if character_data.knowledge_interest[10] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.COMPUTER_SKILLS_INTEREST_IS_LOW)
def handle_computer_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否计算机天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 10 in character_data.knowledge_interest:
        if character_data.knowledge_interest[10] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.PERFORMANCE_SKILLS_INTEREST_IS_HEIGHT)
def handle_performance_interest_is_height(character_id: int) -> int:
    """
    校验角色是否表演天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 11 in character_data.knowledge_interest:
        if character_data.knowledge_interest[11] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.PERFORMANCE_SKILLS_INTEREST_IS_LOW)
def handle_performance_interest_is_low(character_id: int) -> int:
    """
    校验角色是否表演天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 11 in character_data.knowledge_interest:
        if character_data.knowledge_interest[11] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.ELOQUENCE_SKILLS_INTEREST_IS_HEIGHT)
def handle_eloquence_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否口才天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 12 in character_data.knowledge_interest:
        if character_data.knowledge_interest[12] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.ELOQUENCE_SKILLS_INTEREST_IS_LOW)
def handle_eloquence_interest_is_low(character_id: int) -> int:
    """
    校验角色是否口才天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 12 in character_data.knowledge_interest:
        if character_data.knowledge_interest[12] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.PAINTING_SKILLS_INTEREST_IS_HEIGHT)
def handle_painting_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否绘画天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 13 in character_data.knowledge_interest:
        if character_data.knowledge_interest[13] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.PAINTING_SKILLS_INTEREST_IS_LOW)
def handle_painting_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否绘画天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 13 in character_data.knowledge_interest:
        if character_data.knowledge_interest[13] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.SHOOT_SKILLS_INTEREST_IS_HEIGHT)
def handle_ethic_interest_is_height(character_id: int) -> int:
    """
    校验角色是否拍摄天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 14 in character_data.knowledge_interest:
        if character_data.knowledge_interest[14] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.SHOOT_SKILLS_INTEREST_IS_LOW)
def handle_shoot_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否拍摄天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 14 in character_data.knowledge_interest:
        if character_data.knowledge_interest[14] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.SINGING_SKILLS_INTEREST_IS_HEIGHT)
def handle_singing_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否演唱天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 15 in character_data.knowledge_interest:
        if character_data.knowledge_interest[15] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.SINGING_SKILLS_INTEREST_IS_LOW)
def handle_singing_interest_is_low(character_id: int) -> int:
    """
    校验角色是否演唱天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 15 in character_data.knowledge_interest:
        if character_data.knowledge_interest[15] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.WRITE_MUSIC_SKILLS_INTEREST_IS_HEIGHT)
def handle_write_music_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否作曲天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 16 in character_data.knowledge_interest:
        if character_data.knowledge_interest[16] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.WRITE_MUSIC_SKILLS_INTEREST_IS_LOW)
def handle_write_music_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否作曲天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 16 in character_data.knowledge_interest:
        if character_data.knowledge_interest[16] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.COOKING_SKILLS_INTEREST_IS_HEIGHT)
def handle_cooking_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否厨艺天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 17 in character_data.knowledge_interest:
        if character_data.knowledge_interest[17] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.COOKING_SKILLS_INTEREST_IS_LOW)
def handle_cooking_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否厨艺天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 17 in character_data.knowledge_interest:
        if character_data.knowledge_interest[17] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.DANCE_SKILLS_INTEREST_IS_HEIGHT)
def handle_dance_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否舞蹈天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 18 in character_data.knowledge_interest:
        if character_data.knowledge_interest[18] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.DANCE_SKILLS_INTEREST_IS_LOW)
def handle_dance_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否舞蹈天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 18 in character_data.knowledge_interest:
        if character_data.knowledge_interest[18] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.TAILOR_SKILLS_INTEREST_IS_HEIGHT)
def handle_tailor_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否裁缝天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 19 in character_data.knowledge_interest:
        if character_data.knowledge_interest[19] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TAILOR_SKILLS_INTEREST_IS_LOW)
def handle_tailor_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否裁缝天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 19 in character_data.knowledge_interest:
        if character_data.knowledge_interest[19] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.TACTICS_SKILLS_INTEREST_IS_HEIGHT)
def handle_tactics_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否战术天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 20 in character_data.knowledge_interest:
        if character_data.knowledge_interest[20] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TACTICS_SKILLS_INTEREST_IS_LOW)
def handle_tactics_interest_is_low(character_id: int) -> int:
    """
    校验角色是否战术天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 20 in character_data.knowledge_interest:
        if character_data.knowledge_interest[20] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.SWIMMING_SKILLS_INTEREST_IS_HEIGHT)
def handle_swimming_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否游泳天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 21 in character_data.knowledge_interest:
        if character_data.knowledge_interest[21] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.SWIMMING_SKILLS_INTEREST_IS_LOW)
def handle_swimming_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否游泳天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 21 in character_data.knowledge_interest:
        if character_data.knowledge_interest[21] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.MANUFACTURE_SKILLS_INTEREST_IS_HEIGHT)
def handle_manufacture_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否制造天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 22 in character_data.knowledge_interest:
        if character_data.knowledge_interest[22] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.MANUFACTURE_SKILLS_INTEREST_IS_LOW)
def handle_manufacture_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否制造天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 22 in character_data.knowledge_interest:
        if character_data.knowledge_interest[22] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.FIRST_AID_SKILLS_INTEREST_IS_HEIGHT)
def handle_first_aid_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否急救天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 23 in character_data.knowledge_interest:
        if character_data.knowledge_interest[23] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.FIRST_AID_SKILLS_INTEREST_IS_LOW)
def handle_first_aid_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否急救天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 23 in character_data.knowledge_interest:
        if character_data.knowledge_interest[23] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.ANATOMY_SKILLS_INTEREST_IS_HEIGHT)
def handle_anatomy_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否解剖天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 24 in character_data.knowledge_interest:
        if character_data.knowledge_interest[24] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.ANATOMY_SKILLS_INTEREST_IS_LOW)
def handle_anatomy_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否解剖天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 24 in character_data.knowledge_interest:
        if character_data.knowledge_interest[24] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.PLAY_MUSIC_SKILLS_INTEREST_IS_HEIGHT)
def handle_play_music_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否演奏天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 25 in character_data.knowledge_interest:
        if character_data.knowledge_interest[25] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.PLAY_MUSIC_SKILLS_INTEREST_IS_LOW)
def handle_play_music_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否演奏天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 25 in character_data.knowledge_interest:
        if character_data.knowledge_interest[25] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.PROGRAMMING_SKILLS_INTEREST_IS_HEIGHT)
def handle_programming_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否编程天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 26 in character_data.knowledge_interest:
        if character_data.knowledge_interest[26] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.PROGRAMMING_SKILLS_INTEREST_IS_LOW)
def handle_programming_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否编程天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 26 in character_data.knowledge_interest:
        if character_data.knowledge_interest[26] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.HACKER_SKILLS_INTEREST_IS_HEIGHT)
def handle_hacker_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否黑客天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 27 in character_data.knowledge_interest:
        if character_data.knowledge_interest[27] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.HACKER_SKILLS_INTEREST_IS_LOW)
def handle_hacker_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否黑客天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 27 in character_data.knowledge_interest:
        if character_data.knowledge_interest[27] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.WRITE_SKILLS_INTEREST_IS_HEIGHT)
def handle_write_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否写作天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 28 in character_data.knowledge_interest:
        if character_data.knowledge_interest[28] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.WRITE_SKILLS_INTEREST_IS_LOW)
def handle_write_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否写作天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 28 in character_data.knowledge_interest:
        if character_data.knowledge_interest[28] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.TRANSACTION_SKILLS_INTEREST_IS_HEIGHT)
def handle_transction_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否交易天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 29 in character_data.knowledge_interest:
        if character_data.knowledge_interest[29] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TRANSACTION_SKILLS_INTEREST_IS_LOW)
def handle_transction_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否交易天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 29 in character_data.knowledge_interest:
        if character_data.knowledge_interest[29] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.CEREMONY_SKILLS_INTEREST_IS_HEIGHT)
def handle_ceremony_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否礼仪天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 30 in character_data.knowledge_interest:
        if character_data.knowledge_interest[30] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.CEREMONY_SKILLS_INTEREST_IS_LOW)
def handle_ceremony_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否礼仪天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 30 in character_data.knowledge_interest:
        if character_data.knowledge_interest[30] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.MOTION_SKILLS_INTEREST_IS_HEIGHT)
def handle_motion_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否运动天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 31 in character_data.knowledge_interest:
        if character_data.knowledge_interest[31] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.MOTION_SKILLS_INTEREST_IS_LOW)
def handle_motion_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否运动天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 31 in character_data.knowledge_interest:
        if character_data.knowledge_interest[31] < 1:
            return 1
    return 1
