from Script.Design import handle_premise, attr_calculation
from Script.Core import constant, game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.IS_TARGET_FIRST_KISS)
def handle_is_target_first_kiss(character_id: int) -> int:
    """
    校验是否是交互对象的初吻对象
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return character_id == target_data.first_kiss


@handle_premise.add_premise(constant.Premise.SEX_EXPERIENCE_IS_HIGHT)
def handle_sex_experience_is_hight(character_id: int) -> int:
    """
    校验角色是否性技熟练
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.knowledge.setdefault(9, 0)
    return attr_calculation.get_experience_level_weight(character_data.knowledge[9])


@handle_premise.add_premise(constant.Premise.NO_EXPERIENCE_IN_SEX)
def handle_no_experience_in_sex(character_id: int) -> int:
    """
    校验角色是否没有性经验
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    for i in character_data.sex_experience:
        if character_data.sex_experience[i]:
            return 0
    return 1


@handle_premise.add_premise(constant.Premise.RICH_EXPERIENCE_IN_SEX)
def handle_rich_experience_in_sex(character_id: int) -> int:
    """
    校验角色是否性经验丰富
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    now_exp = 0
    for i in character_data.sex_experience:
        now_exp += character_data.sex_experience[i]
    return attr_calculation.get_experience_level_weight(now_exp)


@handle_premise.add_premise(constant.Premise.TARGET_NO_EXPERIENCE_IN_SEX)
def handle_target_no_experience_in_sex(character_id: int) -> int:
    """
    校验交互对象是否没有性经验
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    for i in target_data.sex_experience:
        if target_data.sex_experience[i]:
            return 0
    return 1


@handle_premise.add_premise(constant.Premise.NO_RICH_EXPERIENCE_IN_SEX)
def handle_no_rich_experience_in_sex(character_id: int) -> int:
    """
    校验角色是否性经验不丰富
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    now_exp = 0
    for i in character_data.sex_experience:
        now_exp += character_data.sex_experience[i]
    return 8 - attr_calculation.get_experience_level_weight(now_exp)


@handle_premise.add_premise(constant.Premise.TARGET_NO_FIRST_KISS)
def handle_target_no_first_kiss(character_id: int) -> int:
    """
    校验交互对象是否初吻还在
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.first_kiss == -1


@handle_premise.add_premise(constant.Premise.NO_FIRST_KISS)
def handle_no_first_kiss(character_id: int) -> int:
    """
    校验是否初吻还在
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    return character_data.first_kiss == -1


@handle_premise.add_premise(constant.Premise.TARGET_HAVE_FIRST_KISS)
def handle_target_have_first_kiss(character_id: int) -> int:
    """
    校验交互对象是否初吻不在了
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.first_kiss != -1


@handle_premise.add_premise(constant.Premise.HAVE_FIRST_KISS)
def handle_have_first_kiss(character_id: int) -> int:
    """
    校验是否初吻不在了
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    return character_data.first_kiss != -1


@handle_premise.add_premise(constant.Premise.TARGET_NO_FIRST_HAND_IN_HAND)
def handle_target_no_first_hand_in_hand(character_id: int) -> int:
    """
    校验交互对象是否没有牵过手
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.first_hand_in_hand == -1


@handle_premise.add_premise(constant.Premise.NO_FIRST_HAND_IN_HAND)
def handle_no_first_hand_in_hand(character_id: int) -> int:
    """
    校验是否没有牵过手
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    return character_data.first_hand_in_hand == -1


@handle_premise.add_premise(constant.Premise.HAVE_LIKE_TARGET_NO_FIRST_KISS)
def handle_have_like_target_no_first_kiss(character_id: int) -> int:
    """
    校验是否有自己喜欢的人的初吻还在
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_index = 0
    for i in {4, 5}:
        character_data.social_contact.setdefault(i, set())
        for c in character_data.social_contact[i]:
            c_data: game_type.Character = cache.character_data[c]
            if c_data.first_kiss == -1:
                character_index += 1
    return character_index


@handle_premise.add_premise(constant.Premise.TARGET_CLITORIS_LEVEL_IS_HIGHT)
def handle_target_clitoris_is_hight(character_id: int) -> int:
    """
    校验交互对象是否阴蒂开发度高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.sex_experience.setdefault(2, 0)
    return attr_calculation.get_experience_level_weight(target_data.sex_experience[2])
