import random

from Script.Design import handle_premise, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.KNOW_TARGET_TEMPERAMENT)
def handle_know_target_temperament(character_id: int) -> int:
    """
    校验角色是否了解交互对象的性格
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id == -1:
        return 0
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_TEMPERAMENT not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_TEMPERAMENT]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TARGET_IDENTITY)
def handle_know_target_identity(character_id: int) -> int:
    """
    校验角色是否了解交互对象的身份
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id == -1:
        return 0
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_IDENTITY not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_IDENTITY]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TARGET_GENDER)
def handle_know_target_gender(character_id: int) -> int:
    """
    校验角色是否了解交互对象的性别
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id == -1:
        return 0
    character_data.character_awareness_data.setdefault(target_id, {})
    awareness_data: game_type.CharacterAwareness = None
    if constant.Awareness.KNOW_GENDER not in character_data.character_awareness_data[target_id]:
        awareness_data = game_type.CharacterAwareness()
        awareness_data.cid = constant.Awareness.KNOW_GENDER
        awareness_data.authenticity = True
        awareness_data.got_count = 1
        character_data.character_awareness_data[target_id][constant.Awareness.KNOW_GENDER] = awareness_data
    else:
        awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_GENDER]
        if awareness_data.authenticity:
            awareness_data.got_count += 1
        else:
            awareness_data.got_count -= 1
            if awareness_data.got_count <= 0:
                awareness_data.authenticity = True
                awareness_data.got_count = 1
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TARGET_HUMANITY)
def handle_know_target_humanity(character_id: int) -> int:
    """
    校验角色是否了解交互对象的人文水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id == -1:
        return 0
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_HUMANITY not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_HUMANITY]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TARGET_SKILLS)
def handle_know_target_skills(character_id: int) -> int:
    """
    校验角色是否了解交互对象的技巧水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id == -1:
        return 0
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SKILLS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SKILLS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TARGET_SCIENCE)
def handle_know_target_science(character_id: int) -> int:
    """
    校验角色是否了解交互对象的科学知识水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id == -1:
        return 0
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SCIENCE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SCIENCE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TARGET_MYSTICS)
def handle_know_target_mystics(character_id: int) -> int:
    """
    校验角色是否了解交互对象的神秘学知识水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id == -1:
        return 0
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MYSTICS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MYSTICS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TARGET_LANGUAGE)
def handle_know_target_language(character_id: int) -> int:
    """
    校验角色是否了解交互对象的语言水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id == -1:
        return 0
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_LANGUAGE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LANGUAGE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TARGET_HUMANITY_INTEREST)
def handle_know_target_humanity_interest(character_id: int) -> int:
    """
    校验角色是否了解交互对象的人文天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id == -1:
        return 0
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_HUMANITY_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_HUMANITY_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TARGET_SKILLS_INTEREST)
def handle_know_target_skills_interest(character_id: int) -> int:
    """
    校验角色是否了解交互对象的技巧天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id == -1:
        return 0
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SKILLS_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SKILLS_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TARGET_SCIENCE_INTEREST)
def handle_know_target_science_interest(character_id: int) -> int:
    """
    校验角色是否了解交互对象的科学知识天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id == -1:
        return 0
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SCIENCE_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SCIENCE_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TARGET_MYSTICS_INTEREST)
def handle_know_target_mystics_interest(character_id: int) -> int:
    """
    校验角色是否了解交互对象的神秘学知识天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id == -1:
        return 0
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MYSTICS_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MYSTICS_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TARGET_LANGUAGE_INTEREST)
def handle_know_target_language_interest(character_id: int) -> int:
    """
    校验角色是否了解交互对象的语言天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id == -1:
        return 0
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_LANGUAGE_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LANGUAGE_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TARGET_LIKE_DRESSING_STYLE)
def handle_know_target_like_dressing_style(character_id: int) -> int:
    """
    校验角色是否了解交互对象喜欢的穿搭风格
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id == -1:
        return 0
    if constant.Awareness.KNOW_LIKE_DRESSING_STYLE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LIKE_DRESSING_STYLE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TARGET_MENTAL_STATUS)
def handle_know_target_mental_status(character_id: int) -> int:
    """
    校验角色是否了解交互对象的心灵状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id == -1:
        return 0
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MENTAL_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MENTAL_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TARGET_BODILY_STATUS)
def handle_know_target_bodily_status(character_id: int) -> int:
    """
    校验角色是否了解交互对象的肉体状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id == -1:
        return 0
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_BODILY_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_BODILY_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TARGET_STATURE)
def handle_know_target_stature(character_id: int) -> int:
    """
    校验角色是否了解交互对象的身材
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id == -1:
        return 0
    awareness_data: game_type.CharacterAwareness = None
    if constant.Awareness.KNOW_STATURE not in character_data.character_awareness_data[target_id]:
        awareness_data = game_type.CharacterAwareness()
        awareness_data.cid = constant.Awareness.KNOW_STATURE
        awareness_data.authenticity = True
        awareness_data.got_count = 1
        character_data.character_awareness_data[target_id][constant.Awareness.KNOW_STATURE] = awareness_data
    else:
        awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_STATURE]
        if awareness_data.authenticity:
            awareness_data.got_count += 1
        else:
            awareness_data.got_count -= 1
            if awareness_data.got_count <= 0:
                awareness_data.authenticity = True
                awareness_data.got_count = 1
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TARGET_WEARS)
def handle_know_target_wears(character_id: int) -> int:
    """
    校验角色是否了解交互对象的当前的穿着
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id == -1:
        return 0
    awareness_data: game_type.CharacterAwareness = None
    if constant.Awareness.KNOW_WEARS not in character_data.character_awareness_data[target_id]:
        awareness_data = game_type.CharacterAwareness()
        awareness_data.cid = constant.Awareness.KNOW_WEARS
        awareness_data.authenticity = True
        awareness_data.got_count = 1
        character_data.character_awareness_data[target_id][constant.Awareness.KNOW_WEARS] = awareness_data
    else:
        awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_WEARS]
        if awareness_data.authenticity:
            awareness_data.got_count += 1
        else:
            awareness_data.got_count -= 1
            if awareness_data.got_count <= 0:
                awareness_data.authenticity = True
                awareness_data.got_count = 1
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TARGET_SEX_EXPERIENCE)
def handle_know_target_sex_experience(character_id: int) -> int:
    """
    校验角色是否了解交互对象的性经验
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id == -1:
        return 0
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SEX_EXPERIENCE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SEX_EXPERIENCE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TARGET_SOCIAL_STATUS)
def handle_know_target_social_status(character_id: int) -> int:
    """
    校验角色是否了解交互对象的社交关系
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id == -1:
        return 0
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SOCIAL_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SOCIAL_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TARGET_POSITION)
def handle_know_target_position(character_id: int) -> int:
    """
    校验角色是否了解交互对象所处的地点位置
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id == -1:
        return 0
    awareness_data: game_type.CharacterAwareness = None
    if constant.Awareness.KNOW_POSITION not in character_data.character_awareness_data[target_id]:
        awareness_data = game_type.CharacterAwareness()
        awareness_data.cid = constant.Awareness.KNOW_POSITION
        awareness_data.authenticity = True
        awareness_data.got_count = 1
        character_data.character_awareness_data[target_id][constant.Awareness.KNOW_POSITION] = awareness_data
    else:
        awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_POSITION]
        if awareness_data.authenticity:
            awareness_data.got_count += 1
        else:
            awareness_data.got_count -= 1
            if awareness_data.got_count <= 0:
                awareness_data.authenticity = True
                awareness_data.got_count = 1
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADORE_TARGET_TEMPERAMENT)
def handle_know_adore_target_temperament(character_id: int) -> int:
    """
    校验角色是否了解爱慕的对象的性格
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(10, set())
    if len(character_data.social_contact[10]) == 0:
        return 0
    character_list = list(character_data.social_contact[10])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_TEMPERAMENT not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_TEMPERAMENT]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADORE_TARGET_IDENTITY)
def handle_know_adore_target_identity(character_id: int) -> int:
    """
    校验角色是否了解爱慕的对象的身份
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(10, set())
    if len(character_data.social_contact[10]) == 0:
        return 0
    character_list = list(character_data.social_contact[10])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_IDENTITY not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_IDENTITY]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADORE_TARGET_GENDER)
def handle_know_adore_target_gender(character_id: int) -> int:
    """
    校验角色是否了解爱慕的对象的性别
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(10, set())
    if len(character_data.social_contact[10]) == 0:
        return 0
    character_list = list(character_data.social_contact[10])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_GENDER not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_GENDER]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADORE_TARGET_HUMANITY)
def handle_know_adore_target_humanity(character_id: int) -> int:
    """
    校验角色是否了解爱慕的对象的人文水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(10, set())
    if len(character_data.social_contact[10]) == 0:
        return 0
    character_list = list(character_data.social_contact[10])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_HUMANITY not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_HUMANITY]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADORE_TARGET_SKILLS)
def handle_know_adore_target_skills(character_id: int) -> int:
    """
    校验角色是否了解爱慕的对象的技巧水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(10, set())
    if len(character_data.social_contact[10]) == 0:
        return 0
    character_list = list(character_data.social_contact[10])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SKILLS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SKILLS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADORE_TARGET_SCIENCE)
def handle_know_adore_target_science(character_id: int) -> int:
    """
    校验角色是否了解爱慕的对象的科学知识水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(10, set())
    if len(character_data.social_contact[10]) == 0:
        return 1
    character_list = list(character_data.social_contact[10])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SCIENCE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SCIENCE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADORE_TARGET_MYSTICS)
def handle_know_adore_target_mystics(character_id: int) -> int:
    """
    校验角色是否了解爱慕的对象的神秘学知识水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(10, set())
    if len(character_data.social_contact[10]) == 0:
        return 1
    character_list = list(character_data.social_contact[10])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MYSTICS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MYSTICS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADORE_TARGET_LANGUAGE)
def handle_know_adore_target_language(character_id: int) -> int:
    """
    校验角色是否了解爱慕的对象的语言水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(10, set())
    if len(character_data.social_contact[10]) == 0:
        return 1
    character_list = list(character_data.social_contact[10])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_LANGUAGE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LANGUAGE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADORE_TARGET_HUMANITY_INTEREST)
def handle_know_adore_target_humanity_interest(character_id: int) -> int:
    """
    校验角色是否了解爱慕的对象的人文天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(10, set())
    if len(character_data.social_contact[10]) == 0:
        return 1
    character_list = list(character_data.social_contact[10])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_HUMANITY_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_HUMANITY_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADORE_TARGET_SKILLS_INTEREST)
def handle_know_adore_target_skills_interest(character_id: int) -> int:
    """
    校验角色是否了解爱慕的对象的技巧天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(10, set())
    if len(character_data.social_contact[10]) == 0:
        return 1
    character_list = list(character_data.social_contact[10])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SKILLS_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SKILLS_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADORE_TARGET_SCIENCE_INTEREST)
def handle_know_adore_target_science_interest(character_id: int) -> int:
    """
    校验角色是否了解爱慕的对象的科学知识天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(10, set())
    if len(character_data.social_contact[10]) == 0:
        return 1
    character_list = list(character_data.social_contact[10])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SCIENCE_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SCIENCE_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADORE_TARGET_MYSTICS_INTEREST)
def handle_know_adore_target_mystics_interest(character_id: int) -> int:
    """
    校验角色是否了解爱慕的对象的神秘学知识天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(10, set())
    if len(character_data.social_contact[10]) == 0:
        return 1
    character_list = list(character_data.social_contact[10])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MYSTICS_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MYSTICS_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADORE_TARGET_LANGUAGE_INTEREST)
def handle_know_adore_target_language_interest(character_id: int) -> int:
    """
    校验角色是否了解爱慕的对象的语言天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(10, set())
    if len(character_data.social_contact[10]) == 0:
        return 1
    character_list = list(character_data.social_contact[10])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_LANGUAGE_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LANGUAGE_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADORE_TARGET_LIKE_DRESSING_STYLE)
def handle_know_adore_target_like_dressing_style(character_id: int) -> int:
    """
    校验角色是否了解爱慕的对象喜欢的穿搭风格
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(10, set())
    if len(character_data.social_contact[10]) == 0:
        return 1
    character_list = list(character_data.social_contact[10])
    target_id = random.choice(character_list)
    if constant.Awareness.KNOW_LIKE_DRESSING_STYLE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LIKE_DRESSING_STYLE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADORE_TARGET_MENTAL_STATUS)
def handle_know_adore_target_mental_status(character_id: int) -> int:
    """
    校验角色是否了解爱慕的对象的心灵状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(10, set())
    if len(character_data.social_contact[10]) == 0:
        return 1
    character_list = list(character_data.social_contact[10])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MENTAL_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MENTAL_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADORE_TARGET_BODILY_STATUS)
def handle_know_adore_target_bodily_status(character_id: int) -> int:
    """
    校验角色是否了解爱慕的对象的肉体状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(10, set())
    if len(character_data.social_contact[10]) == 0:
        return 1
    character_list = list(character_data.social_contact[10])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_BODILY_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_BODILY_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADORE_TARGET_STATURE)
def handle_know_adore_target_stature(character_id: int) -> int:
    """
    校验角色是否了解爱慕的对象的身材
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(10, set())
    if len(character_data.social_contact[10]) == 0:
        return 1
    character_list = list(character_data.social_contact[10])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_STATURE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_ADORE_TARGET_STATURE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADORE_TARGET_WEARS)
def handle_know_adore_target_wears(character_id: int) -> int:
    """
    校验角色是否了解爱慕的对象的当前的穿着
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(10, set())
    if len(character_data.social_contact[10]) == 0:
        return 1
    character_list = list(character_data.social_contact[10])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_ADORE_TARGET_WEARS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_ADORE_TARGET_WEARS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1



@handle_premise.add_premise(constant.Premise.KNOW_ADORE_TARGET_SEX_EXPERIENCE)
def handle_know_adore_target_sex_experience(character_id: int) -> int:
    """
    校验角色是否了解爱慕的对象的性经验
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(10, set())
    if len(character_data.social_contact[10]) == 0:
        return 1
    character_list = list(character_data.social_contact[10])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_ADORE_TARGET_SEX_EXPERIENCE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_ADORE_TARGET_SEX_EXPERIENCE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADORE_TARGET_SOCIAL_STATUS)
def handle_know_adore_target_social_status(character_id: int) -> int:
    """
    校验角色是否了解爱慕的对象的社交关系
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(10, set())
    if len(character_data.social_contact[10]) == 0:
        return 1
    character_list = list(character_data.social_contact[10])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SOCIAL_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SOCIAL_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADORE_TARGET_POSITION)
def handle_know_adore_target_position(character_id: int) -> int:
    """
    校验角色是否了解爱慕的对象所处的地点位置
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(10, set())
    if len(character_data.social_contact[10]) == 0:
        return 1
    character_list = list(character_data.social_contact[10])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_POSITION not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_POSITION]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADMIRE_TARGET_TEMPERAMENT)
def handle_know_admire_target_temperament(character_id: int) -> int:
    """
    校验角色是否了解恋慕的对象的性格
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(9, set())
    if len(character_data.social_contact[9]) == 0:
        return 0
    character_list = list(character_data.social_contact[9])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_TEMPERAMENT not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_TEMPERAMENT]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADMIRE_TARGET_IDENTITY)
def handle_know_admire_target_identity(character_id: int) -> int:
    """
    校验角色是否了解恋慕的对象的身份
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(9, set())
    if len(character_data.social_contact[9]) == 0:
        return 0
    character_list = list(character_data.social_contact[9])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_IDENTITY not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_IDENTITY]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADMIRE_TARGET_GENDER)
def handle_know_admire_target_gender(character_id: int) -> int:
    """
    校验角色是否了解恋慕的对象的性别
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(9, set())
    if len(character_data.social_contact[9]) == 0:
        return 0
    character_list = list(character_data.social_contact[9])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_GENDER not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_GENDER]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADMIRE_TARGET_HUMANITY)
def handle_know_admire_target_humanity(character_id: int) -> int:
    """
    校验角色是否了解恋慕的对象的人文水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(9, set())
    if len(character_data.social_contact[9]) == 0:
        return 0
    character_list = list(character_data.social_contact[9])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_HUMANITY not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_HUMANITY]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADMIRE_TARGET_SKILLS)
def handle_know_admire_target_skills(character_id: int) -> int:
    """
    校验角色是否了解恋慕的对象的技巧水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(9, set())
    if len(character_data.social_contact[9]) == 0:
        return 0
    character_list = list(character_data.social_contact[9])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SKILLS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SKILLS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADMIRE_TARGET_SCIENCE)
def handle_know_admire_target_science(character_id: int) -> int:
    """
    校验角色是否了解恋慕的对象的科学知识水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(9, set())
    if len(character_data.social_contact[9]) == 0:
        return 1
    character_list = list(character_data.social_contact[9])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SCIENCE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SCIENCE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADMIRE_TARGET_MYSTICS)
def handle_know_admire_target_mystics(character_id: int) -> int:
    """
    校验角色是否了解恋慕的对象的神秘学知识水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(9, set())
    if len(character_data.social_contact[9]) == 0:
        return 1
    character_list = list(character_data.social_contact[9])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MYSTICS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MYSTICS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADMIRE_TARGET_LANGUAGE)
def handle_know_admire_target_language(character_id: int) -> int:
    """
    校验角色是否了解恋慕的对象的语言水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(9, set())
    if len(character_data.social_contact[9]) == 0:
        return 1
    character_list = list(character_data.social_contact[9])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_LANGUAGE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LANGUAGE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADMIRE_TARGET_HUMANITY_INTEREST)
def handle_know_admire_target_humanity_interest(character_id: int) -> int:
    """
    校验角色是否了解恋慕的对象的人文天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(9, set())
    if len(character_data.social_contact[9]) == 0:
        return 1
    character_list = list(character_data.social_contact[9])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_HUMANITY_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_HUMANITY_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADMIRE_TARGET_SKILLS_INTEREST)
def handle_know_admire_target_skills_interest(character_id: int) -> int:
    """
    校验角色是否了解恋慕的对象的技巧天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(9, set())
    if len(character_data.social_contact[9]) == 0:
        return 1
    character_list = list(character_data.social_contact[9])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SKILLS_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SKILLS_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADMIRE_TARGET_SCIENCE_INTEREST)
def handle_know_admire_target_science_interest(character_id: int) -> int:
    """
    校验角色是否了解恋慕的对象的科学知识天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(9, set())
    if len(character_data.social_contact[9]) == 0:
        return 1
    character_list = list(character_data.social_contact[9])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SCIENCE_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SCIENCE_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADMIRE_TARGET_MYSTICS_INTEREST)
def handle_know_admire_target_mystics_interest(character_id: int) -> int:
    """
    校验角色是否了解恋慕的对象的神秘学知识天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(9, set())
    if len(character_data.social_contact[9]) == 0:
        return 1
    character_list = list(character_data.social_contact[9])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MYSTICS_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MYSTICS_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADMIRE_TARGET_LANGUAGE_INTEREST)
def handle_know_admire_target_language_interest(character_id: int) -> int:
    """
    校验角色是否了解恋慕的对象的语言天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(9, set())
    if len(character_data.social_contact[9]) == 0:
        return 1
    character_list = list(character_data.social_contact[9])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_LANGUAGE_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LANGUAGE_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADMIRE_TARGET_LIKE_DRESSING_STYLE)
def handle_know_admire_target_like_dressing_style(character_id: int) -> int:
    """
    校验角色是否了解恋慕的对象喜欢的穿搭风格
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(9, set())
    if len(character_data.social_contact[9]) == 0:
        return 1
    character_list = list(character_data.social_contact[9])
    target_id = random.choice(character_list)
    if constant.Awareness.KNOW_LIKE_DRESSING_STYLE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LIKE_DRESSING_STYLE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADMIRE_TARGET_MENTAL_STATUS)
def handle_know_admire_target_mental_status(character_id: int) -> int:
    """
    校验角色是否了解恋慕的对象的心灵状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(9, set())
    if len(character_data.social_contact[9]) == 0:
        return 1
    character_list = list(character_data.social_contact[9])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MENTAL_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MENTAL_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADMIRE_TARGET_BODILY_STATUS)
def handle_know_admire_target_bodily_status(character_id: int) -> int:
    """
    校验角色是否了解恋慕的对象的肉体状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(9, set())
    if len(character_data.social_contact[9]) == 0:
        return 1
    character_list = list(character_data.social_contact[9])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_BODILY_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_BODILY_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADMIRE_TARGET_STATURE)
def handle_know_admire_target_stature(character_id: int) -> int:
    """
    校验角色是否了解恋慕的对象的身材
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(9, set())
    if len(character_data.social_contact[9]) == 0:
        return 1
    character_list = list(character_data.social_contact[9])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_STATURE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_ADORE_TARGET_STATURE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADMIRE_TARGET_WEARS)
def handle_know_admire_target_wears(character_id: int) -> int:
    """
    校验角色是否了解恋慕的对象的当前的穿着
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(9, set())
    if len(character_data.social_contact[9]) == 0:
        return 1
    character_list = list(character_data.social_contact[9])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_ADORE_TARGET_WEARS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_ADORE_TARGET_WEARS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1



@handle_premise.add_premise(constant.Premise.KNOW_ADMIRE_TARGET_SEX_EXPERIENCE)
def handle_know_admire_target_sex_experience(character_id: int) -> int:
    """
    校验角色是否了解恋慕的对象的性经验
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(9, set())
    if len(character_data.social_contact[9]) == 0:
        return 1
    character_list = list(character_data.social_contact[9])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_ADORE_TARGET_SEX_EXPERIENCE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_ADORE_TARGET_SEX_EXPERIENCE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADMIRE_TARGET_SOCIAL_STATUS)
def handle_know_admire_target_social_status(character_id: int) -> int:
    """
    校验角色是否了解恋慕的对象的社交关系
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(9, set())
    if len(character_data.social_contact[9]) == 0:
        return 1
    character_list = list(character_data.social_contact[9])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SOCIAL_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SOCIAL_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_ADMIRE_TARGET_POSITION)
def handle_know_admire_target_position(character_id: int) -> int:
    """
    校验角色是否了解恋慕的对象所处的地点位置
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(9, set())
    if len(character_data.social_contact[9]) == 0:
        return 1
    character_list = list(character_data.social_contact[9])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_POSITION not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_POSITION]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1

@handle_premise.add_premise(constant.Premise.KNOW_DEPEND_ON_TARGET_TEMPERAMENT)
def handle_know_depend_on_target_temperament(character_id: int) -> int:
    """
    校验角色是否了解依靠的对象的性格
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(8, set())
    if len(character_data.social_contact[8]) == 0:
        return 0
    character_list = list(character_data.social_contact[8])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_TEMPERAMENT not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_TEMPERAMENT]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEPEND_ON_TARGET_IDENTITY)
def handle_know_depend_on_target_identity(character_id: int) -> int:
    """
    校验角色是否了解依靠的对象的身份
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(8, set())
    if len(character_data.social_contact[8]) == 0:
        return 0
    character_list = list(character_data.social_contact[8])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_IDENTITY not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_IDENTITY]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEPEND_ON_TARGET_GENDER)
def handle_know_depend_on_target_gender(character_id: int) -> int:
    """
    校验角色是否了解依靠的对象的性别
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(8, set())
    if len(character_data.social_contact[8]) == 0:
        return 0
    character_list = list(character_data.social_contact[8])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_GENDER not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_GENDER]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEPEND_ON_TARGET_HUMANITY)
def handle_know_depend_on_target_humanity(character_id: int) -> int:
    """
    校验角色是否了解依靠的对象的人文水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(8, set())
    if len(character_data.social_contact[8]) == 0:
        return 0
    character_list = list(character_data.social_contact[8])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_HUMANITY not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_HUMANITY]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEPEND_ON_TARGET_SKILLS)
def handle_know_depend_on_target_skills(character_id: int) -> int:
    """
    校验角色是否了解依靠的对象的技巧水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(8, set())
    if len(character_data.social_contact[8]) == 0:
        return 0
    character_list = list(character_data.social_contact[8])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SKILLS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SKILLS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEPEND_ON_TARGET_SCIENCE)
def handle_know_depend_on_target_science(character_id: int) -> int:
    """
    校验角色是否了解依靠的对象的科学知识水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(8, set())
    if len(character_data.social_contact[8]) == 0:
        return 1
    character_list = list(character_data.social_contact[8])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SCIENCE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SCIENCE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEPEND_ON_TARGET_MYSTICS)
def handle_know_depend_on_target_mystics(character_id: int) -> int:
    """
    校验角色是否了解依靠的对象的神秘学知识水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(8, set())
    if len(character_data.social_contact[8]) == 0:
        return 1
    character_list = list(character_data.social_contact[8])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MYSTICS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MYSTICS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEPEND_ON_TARGET_LANGUAGE)
def handle_know_depend_on_target_language(character_id: int) -> int:
    """
    校验角色是否了解依靠的对象的语言水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(8, set())
    if len(character_data.social_contact[8]) == 0:
        return 1
    character_list = list(character_data.social_contact[8])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_LANGUAGE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LANGUAGE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEPEND_ON_TARGET_HUMANITY_INTEREST)
def handle_know_depend_on_target_humanity_interest(character_id: int) -> int:
    """
    校验角色是否了解依靠的对象的人文天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(8, set())
    if len(character_data.social_contact[8]) == 0:
        return 1
    character_list = list(character_data.social_contact[8])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_HUMANITY_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_HUMANITY_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEPEND_ON_TARGET_SKILLS_INTEREST)
def handle_know_depend_on_target_skills_interest(character_id: int) -> int:
    """
    校验角色是否了解依靠的对象的技巧天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(8, set())
    if len(character_data.social_contact[8]) == 0:
        return 1
    character_list = list(character_data.social_contact[8])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SKILLS_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SKILLS_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEPEND_ON_TARGET_SCIENCE_INTEREST)
def handle_know_depend_on_target_science_interest(character_id: int) -> int:
    """
    校验角色是否了解依靠的对象的科学知识天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(8, set())
    if len(character_data.social_contact[8]) == 0:
        return 1
    character_list = list(character_data.social_contact[8])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SCIENCE_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SCIENCE_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEPEND_ON_TARGET_MYSTICS_INTEREST)
def handle_know_depend_on_target_mystics_interest(character_id: int) -> int:
    """
    校验角色是否了解依靠的对象的神秘学知识天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(8, set())
    if len(character_data.social_contact[8]) == 0:
        return 1
    character_list = list(character_data.social_contact[8])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MYSTICS_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MYSTICS_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEPEND_ON_TARGET_LANGUAGE_INTEREST)
def handle_know_depend_on_target_language_interest(character_id: int) -> int:
    """
    校验角色是否了解依靠的对象的语言天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(8, set())
    if len(character_data.social_contact[8]) == 0:
        return 1
    character_list = list(character_data.social_contact[8])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_LANGUAGE_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LANGUAGE_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEPEND_ON_TARGET_LIKE_DRESSING_STYLE)
def handle_know_depend_on_target_like_dressing_style(character_id: int) -> int:
    """
    校验角色是否了解依靠的对象喜欢的穿搭风格
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(8, set())
    if len(character_data.social_contact[8]) == 0:
        return 1
    character_list = list(character_data.social_contact[8])
    target_id = random.choice(character_list)
    if constant.Awareness.KNOW_LIKE_DRESSING_STYLE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LIKE_DRESSING_STYLE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEPEND_ON_TARGET_MENTAL_STATUS)
def handle_know_depend_on_target_mental_status(character_id: int) -> int:
    """
    校验角色是否了解依靠的对象的心灵状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(8, set())
    if len(character_data.social_contact[8]) == 0:
        return 1
    character_list = list(character_data.social_contact[8])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MENTAL_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MENTAL_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEPEND_ON_TARGET_BODILY_STATUS)
def handle_know_depend_on_target_bodily_status(character_id: int) -> int:
    """
    校验角色是否了解依靠的对象的肉体状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(8, set())
    if len(character_data.social_contact[8]) == 0:
        return 1
    character_list = list(character_data.social_contact[8])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_BODILY_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_BODILY_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEPEND_ON_TARGET_STATURE)
def handle_know_depend_on_target_stature(character_id: int) -> int:
    """
    校验角色是否了解依靠的对象的身材
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(8, set())
    if len(character_data.social_contact[8]) == 0:
        return 1
    character_list = list(character_data.social_contact[8])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_STATURE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_ADORE_TARGET_STATURE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEPEND_ON_TARGET_WEARS)
def handle_know_depend_on_target_wears(character_id: int) -> int:
    """
    校验角色是否了解依靠的对象的当前的穿着
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(8, set())
    if len(character_data.social_contact[8]) == 0:
        return 1
    character_list = list(character_data.social_contact[8])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_ADORE_TARGET_WEARS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_ADORE_TARGET_WEARS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1



@handle_premise.add_premise(constant.Premise.KNOW_DEPEND_ON_TARGET_SEX_EXPERIENCE)
def handle_know_depend_on_target_sex_experience(character_id: int) -> int:
    """
    校验角色是否了解依靠的对象的性经验
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(8, set())
    if len(character_data.social_contact[8]) == 0:
        return 1
    character_list = list(character_data.social_contact[8])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_ADORE_TARGET_SEX_EXPERIENCE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_ADORE_TARGET_SEX_EXPERIENCE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEPEND_ON_TARGET_SOCIAL_STATUS)
def handle_know_depend_on_target_social_status(character_id: int) -> int:
    """
    校验角色是否了解依靠的对象的社交关系
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(8, set())
    if len(character_data.social_contact[8]) == 0:
        return 1
    character_list = list(character_data.social_contact[8])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SOCIAL_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SOCIAL_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEPEND_ON_TARGET_POSITION)
def handle_know_depend_on_target_position(character_id: int) -> int:
    """
    校验角色是否了解依靠的对象所处的地点位置
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(8, set())
    if len(character_data.social_contact[8]) == 0:
        return 1
    character_list = list(character_data.social_contact[8])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_POSITION not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_POSITION]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TRUST_TARGET_TEMPERAMENT)
def handle_know_trust_target_temperament(character_id: int) -> int:
    """
    校验角色是否了解信任的对象的性格
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(7, set())
    if len(character_data.social_contact[7]) == 0:
        return 0
    character_list = list(character_data.social_contact[7])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_TEMPERAMENT not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_TEMPERAMENT]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TRUST_TARGET_IDENTITY)
def handle_know_trust_target_identity(character_id: int) -> int:
    """
    校验角色是否了解信任的对象的身份
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(7, set())
    if len(character_data.social_contact[7]) == 0:
        return 0
    character_list = list(character_data.social_contact[7])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_IDENTITY not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_IDENTITY]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TRUST_TARGET_GENDER)
def handle_know_trust_target_gender(character_id: int) -> int:
    """
    校验角色是否了解信任的对象的性别
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(7, set())
    if len(character_data.social_contact[7]) == 0:
        return 0
    character_list = list(character_data.social_contact[7])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_GENDER not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_GENDER]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TRUST_TARGET_HUMANITY)
def handle_know_trust_target_humanity(character_id: int) -> int:
    """
    校验角色是否了解信任的对象的人文水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(7, set())
    if len(character_data.social_contact[7]) == 0:
        return 0
    character_list = list(character_data.social_contact[7])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_HUMANITY not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_HUMANITY]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TRUST_TARGET_SKILLS)
def handle_know_trust_target_skills(character_id: int) -> int:
    """
    校验角色是否了解信任的对象的技巧水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(7, set())
    if len(character_data.social_contact[7]) == 0:
        return 0
    character_list = list(character_data.social_contact[7])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SKILLS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SKILLS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TRUST_TARGET_SCIENCE)
def handle_know_trust_target_science(character_id: int) -> int:
    """
    校验角色是否了解信任的对象的科学知识水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(7, set())
    if len(character_data.social_contact[7]) == 0:
        return 1
    character_list = list(character_data.social_contact[7])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SCIENCE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SCIENCE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TRUST_TARGET_MYSTICS)
def handle_know_trust_target_mystics(character_id: int) -> int:
    """
    校验角色是否了解信任的对象的神秘学知识水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(7, set())
    if len(character_data.social_contact[7]) == 0:
        return 1
    character_list = list(character_data.social_contact[7])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MYSTICS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MYSTICS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TRUST_TARGET_LANGUAGE)
def handle_know_trust_target_language(character_id: int) -> int:
    """
    校验角色是否了解信任的对象的语言水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(7, set())
    if len(character_data.social_contact[7]) == 0:
        return 1
    character_list = list(character_data.social_contact[7])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_LANGUAGE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LANGUAGE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TRUST_TARGET_HUMANITY_INTEREST)
def handle_know_trust_target_humanity_interest(character_id: int) -> int:
    """
    校验角色是否了解信任的对象的人文天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(7, set())
    if len(character_data.social_contact[7]) == 0:
        return 1
    character_list = list(character_data.social_contact[7])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_HUMANITY_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_HUMANITY_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TRUST_TARGET_SKILLS_INTEREST)
def handle_know_trust_target_skills_interest(character_id: int) -> int:
    """
    校验角色是否了解信任的对象的技巧天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(7, set())
    if len(character_data.social_contact[7]) == 0:
        return 1
    character_list = list(character_data.social_contact[7])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SKILLS_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SKILLS_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TRUST_TARGET_SCIENCE_INTEREST)
def handle_know_trust_target_science_interest(character_id: int) -> int:
    """
    校验角色是否了解信任的对象的科学知识天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(7, set())
    if len(character_data.social_contact[7]) == 0:
        return 1
    character_list = list(character_data.social_contact[7])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SCIENCE_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SCIENCE_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TRUST_TARGET_MYSTICS_INTEREST)
def handle_know_trust_target_mystics_interest(character_id: int) -> int:
    """
    校验角色是否了解信任的对象的神秘学知识天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(7, set())
    if len(character_data.social_contact[7]) == 0:
        return 1
    character_list = list(character_data.social_contact[7])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MYSTICS_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MYSTICS_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TRUST_TARGET_LANGUAGE_INTEREST)
def handle_know_trust_target_language_interest(character_id: int) -> int:
    """
    校验角色是否了解信任的对象的语言天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(7, set())
    if len(character_data.social_contact[7]) == 0:
        return 1
    character_list = list(character_data.social_contact[7])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_LANGUAGE_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LANGUAGE_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TRUST_TARGET_LIKE_DRESSING_STYLE)
def handle_know_trust_target_like_dressing_style(character_id: int) -> int:
    """
    校验角色是否了解信任的对象喜欢的穿搭风格
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(7, set())
    if len(character_data.social_contact[7]) == 0:
        return 1
    character_list = list(character_data.social_contact[7])
    target_id = random.choice(character_list)
    if constant.Awareness.KNOW_LIKE_DRESSING_STYLE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LIKE_DRESSING_STYLE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TRUST_TARGET_MENTAL_STATUS)
def handle_know_trust_target_mental_status(character_id: int) -> int:
    """
    校验角色是否了解信任的对象的心灵状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(7, set())
    if len(character_data.social_contact[7]) == 0:
        return 1
    character_list = list(character_data.social_contact[7])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MENTAL_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MENTAL_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TRUST_TARGET_BODILY_STATUS)
def handle_know_trust_target_bodily_status(character_id: int) -> int:
    """
    校验角色是否了解信任的对象的肉体状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(7, set())
    if len(character_data.social_contact[7]) == 0:
        return 1
    character_list = list(character_data.social_contact[7])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_BODILY_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_BODILY_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TRUST_TARGET_STATURE)
def handle_know_trust_target_stature(character_id: int) -> int:
    """
    校验角色是否了解信任的对象的身材
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(7, set())
    if len(character_data.social_contact[7]) == 0:
        return 1
    character_list = list(character_data.social_contact[7])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_STATURE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_ADORE_TARGET_STATURE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TRUST_TARGET_WEARS)
def handle_know_trust_target_wears(character_id: int) -> int:
    """
    校验角色是否了解信任的对象的当前的穿着
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(7, set())
    if len(character_data.social_contact[7]) == 0:
        return 1
    character_list = list(character_data.social_contact[7])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_ADORE_TARGET_WEARS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_ADORE_TARGET_WEARS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1



@handle_premise.add_premise(constant.Premise.KNOW_TRUST_TARGET_SEX_EXPERIENCE)
def handle_know_trust_target_sex_experience(character_id: int) -> int:
    """
    校验角色是否了解信任的对象的性经验
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(7, set())
    if len(character_data.social_contact[7]) == 0:
        return 1
    character_list = list(character_data.social_contact[7])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_ADORE_TARGET_SEX_EXPERIENCE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_ADORE_TARGET_SEX_EXPERIENCE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TRUST_TARGET_SOCIAL_STATUS)
def handle_know_trust_target_social_status(character_id: int) -> int:
    """
    校验角色是否了解信任的对象的社交关系
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(7, set())
    if len(character_data.social_contact[7]) == 0:
        return 1
    character_list = list(character_data.social_contact[7])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SOCIAL_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SOCIAL_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_TRUST_TARGET_POSITION)
def handle_know_trust_target_position(character_id: int) -> int:
    """
    校验角色是否了解信任的对象所处的地点位置
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(7, set())
    if len(character_data.social_contact[7]) == 0:
        return 1
    character_list = list(character_data.social_contact[7])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_POSITION not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_POSITION]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_FRIEND_TEMPERAMENT)
def handle_know_friend_temperament(character_id: int) -> int:
    """
    校验角色是否了解朋友的性格
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(6, set())
    if len(character_data.social_contact[6]) == 0:
        return 0
    character_list = list(character_data.social_contact[6])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_TEMPERAMENT not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_TEMPERAMENT]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_FRIEND_IDENTITY)
def handle_know_friend_identity(character_id: int) -> int:
    """
    校验角色是否了解朋友的身份
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(6, set())
    if len(character_data.social_contact[6]) == 0:
        return 0
    character_list = list(character_data.social_contact[6])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_IDENTITY not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_IDENTITY]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_FRIEND_GENDER)
def handle_know_friend_gender(character_id: int) -> int:
    """
    校验角色是否了解朋友的性别
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(6, set())
    if len(character_data.social_contact[6]) == 0:
        return 0
    character_list = list(character_data.social_contact[6])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_GENDER not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_GENDER]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_FRIEND_HUMANITY)
def handle_know_friend_humanity(character_id: int) -> int:
    """
    校验角色是否了解朋友的人文水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(6, set())
    if len(character_data.social_contact[6]) == 0:
        return 0
    character_list = list(character_data.social_contact[6])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_HUMANITY not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_HUMANITY]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_FRIEND_SKILLS)
def handle_know_friend_skills(character_id: int) -> int:
    """
    校验角色是否了解朋友的技巧水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(6, set())
    if len(character_data.social_contact[6]) == 0:
        return 0
    character_list = list(character_data.social_contact[6])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SKILLS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SKILLS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_FRIEND_SCIENCE)
def handle_know_friend_science(character_id: int) -> int:
    """
    校验角色是否了解朋友的科学知识水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(6, set())
    if len(character_data.social_contact[6]) == 0:
        return 1
    character_list = list(character_data.social_contact[6])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SCIENCE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SCIENCE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_FRIEND_MYSTICS)
def handle_know_friend_mystics(character_id: int) -> int:
    """
    校验角色是否了解朋友的神秘学知识水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(6, set())
    if len(character_data.social_contact[6]) == 0:
        return 1
    character_list = list(character_data.social_contact[6])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MYSTICS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MYSTICS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_FRIEND_LANGUAGE)
def handle_know_friend_language(character_id: int) -> int:
    """
    校验角色是否了解朋友的语言水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(6, set())
    if len(character_data.social_contact[6]) == 0:
        return 1
    character_list = list(character_data.social_contact[6])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_LANGUAGE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LANGUAGE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_FRIEND_HUMANITY_INTEREST)
def handle_know_friend_humanity_interest(character_id: int) -> int:
    """
    校验角色是否了解朋友的人文天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(6, set())
    if len(character_data.social_contact[6]) == 0:
        return 1
    character_list = list(character_data.social_contact[6])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_HUMANITY_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_HUMANITY_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_FRIEND_SKILLS_INTEREST)
def handle_know_friend_skills_interest(character_id: int) -> int:
    """
    校验角色是否了解朋友的技巧天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(6, set())
    if len(character_data.social_contact[6]) == 0:
        return 1
    character_list = list(character_data.social_contact[6])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SKILLS_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SKILLS_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_FRIEND_SCIENCE_INTEREST)
def handle_know_friend_science_interest(character_id: int) -> int:
    """
    校验角色是否了解朋友的科学知识天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(6, set())
    if len(character_data.social_contact[6]) == 0:
        return 1
    character_list = list(character_data.social_contact[6])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SCIENCE_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SCIENCE_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_FRIEND_MYSTICS_INTEREST)
def handle_know_friend_mystics_interest(character_id: int) -> int:
    """
    校验角色是否了解朋友的神秘学知识天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(6, set())
    if len(character_data.social_contact[6]) == 0:
        return 1
    character_list = list(character_data.social_contact[6])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MYSTICS_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MYSTICS_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_FRIEND_LANGUAGE_INTEREST)
def handle_know_friend_language_interest(character_id: int) -> int:
    """
    校验角色是否了解朋友的语言天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(6, set())
    if len(character_data.social_contact[6]) == 0:
        return 1
    character_list = list(character_data.social_contact[6])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_LANGUAGE_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LANGUAGE_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_FRIEND_LIKE_DRESSING_STYLE)
def handle_know_friend_like_dressing_style(character_id: int) -> int:
    """
    校验角色是否了解朋友喜欢的穿搭风格
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(6, set())
    if len(character_data.social_contact[6]) == 0:
        return 1
    character_list = list(character_data.social_contact[6])
    target_id = random.choice(character_list)
    if constant.Awareness.KNOW_LIKE_DRESSING_STYLE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LIKE_DRESSING_STYLE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_FRIEND_MENTAL_STATUS)
def handle_know_friend_mental_status(character_id: int) -> int:
    """
    校验角色是否了解朋友的心灵状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(6, set())
    if len(character_data.social_contact[6]) == 0:
        return 1
    character_list = list(character_data.social_contact[6])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MENTAL_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MENTAL_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_FRIEND_BODILY_STATUS)
def handle_know_friend_bodily_status(character_id: int) -> int:
    """
    校验角色是否了解朋友的肉体状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(6, set())
    if len(character_data.social_contact[6]) == 0:
        return 1
    character_list = list(character_data.social_contact[6])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_BODILY_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_BODILY_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_FRIEND_STATURE)
def handle_know_friend_stature(character_id: int) -> int:
    """
    校验角色是否了解朋友的身材
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(6, set())
    if len(character_data.social_contact[6]) == 0:
        return 1
    character_list = list(character_data.social_contact[6])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_STATURE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_ADORE_TARGET_STATURE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_FRIEND_WEARS)
def handle_know_friend_wears(character_id: int) -> int:
    """
    校验角色是否了解朋友的当前的穿着
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(6, set())
    if len(character_data.social_contact[6]) == 0:
        return 1
    character_list = list(character_data.social_contact[6])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_ADORE_TARGET_WEARS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_ADORE_TARGET_WEARS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1



@handle_premise.add_premise(constant.Premise.KNOW_FRIEND_SEX_EXPERIENCE)
def handle_know_friend_sex_experience(character_id: int) -> int:
    """
    校验角色是否了解朋友的性经验
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(6, set())
    if len(character_data.social_contact[6]) == 0:
        return 1
    character_list = list(character_data.social_contact[6])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_ADORE_TARGET_SEX_EXPERIENCE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_ADORE_TARGET_SEX_EXPERIENCE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_FRIEND_SOCIAL_STATUS)
def handle_know_friend_social_status(character_id: int) -> int:
    """
    校验角色是否了解朋友的社交关系
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(6, set())
    if len(character_data.social_contact[6]) == 0:
        return 1
    character_list = list(character_data.social_contact[6])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SOCIAL_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SOCIAL_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_FRIEND_TARGET_POSITION)
def handle_know_friend_position(character_id: int) -> int:
    """
    校验角色是否了解朋友所处的地点位置
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(6, set())
    if len(character_data.social_contact[6]) == 0:
        return 1
    character_list = list(character_data.social_contact[6])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_POSITION not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_POSITION]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DISLIKE_TARGET_TEMPERAMENT)
def handle_know_dislike_target_temperament(character_id: int) -> int:
    """
    校验角色是否了解讨厌的人的性格
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(4, set())
    if len(character_data.social_contact[4]) == 0:
        return 0
    character_list = list(character_data.social_contact[4])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_TEMPERAMENT not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_TEMPERAMENT]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DISLIKE_TARGET_IDENTITY)
def handle_know_dislike_target_identity(character_id: int) -> int:
    """
    校验角色是否了解讨厌的人的身份
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(4, set())
    if len(character_data.social_contact[4]) == 0:
        return 0
    character_list = list(character_data.social_contact[4])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_IDENTITY not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_IDENTITY]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DISLIKE_TARGET_GENDER)
def handle_know_dislike_target_gender(character_id: int) -> int:
    """
    校验角色是否了解讨厌的人的性别
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(4, set())
    if len(character_data.social_contact[4]) == 0:
        return 0
    character_list = list(character_data.social_contact[4])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_GENDER not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_GENDER]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DISLIKE_TARGET_HUMANITY)
def handle_know_dislike_target_humanity(character_id: int) -> int:
    """
    校验角色是否了解讨厌的人的人文水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(4, set())
    if len(character_data.social_contact[4]) == 0:
        return 0
    character_list = list(character_data.social_contact[4])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_HUMANITY not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_HUMANITY]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DISLIKE_TARGET_SKILLS)
def handle_know_dislike_target_skills(character_id: int) -> int:
    """
    校验角色是否了解讨厌的人的技巧水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(4, set())
    if len(character_data.social_contact[4]) == 0:
        return 0
    character_list = list(character_data.social_contact[4])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SKILLS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SKILLS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DISLIKE_TARGET_SCIENCE)
def handle_know_dislike_target_science(character_id: int) -> int:
    """
    校验角色是否了解讨厌的人的科学知识水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(4, set())
    if len(character_data.social_contact[4]) == 0:
        return 1
    character_list = list(character_data.social_contact[4])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SCIENCE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SCIENCE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DISLIKE_TARGET_MYSTICS)
def handle_know_dislike_target_mystics(character_id: int) -> int:
    """
    校验角色是否了解讨厌的人的神秘学知识水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(4, set())
    if len(character_data.social_contact[4]) == 0:
        return 1
    character_list = list(character_data.social_contact[4])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MYSTICS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MYSTICS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DISLIKE_TARGET_LANGUAGE)
def handle_know_dislike_target_language(character_id: int) -> int:
    """
    校验角色是否了解讨厌的人的语言水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(4, set())
    if len(character_data.social_contact[4]) == 0:
        return 1
    character_list = list(character_data.social_contact[4])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_LANGUAGE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LANGUAGE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DISLIKE_TARGET_HUMANITY_INTEREST)
def handle_know_dislike_target_humanity_interest(character_id: int) -> int:
    """
    校验角色是否了解讨厌的人的人文天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(4, set())
    if len(character_data.social_contact[4]) == 0:
        return 1
    character_list = list(character_data.social_contact[4])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_HUMANITY_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_HUMANITY_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DISLIKE_TARGET_SKILLS_INTEREST)
def handle_know_dislike_target_skills_interest(character_id: int) -> int:
    """
    校验角色是否了解讨厌的人的技巧天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(4, set())
    if len(character_data.social_contact[4]) == 0:
        return 1
    character_list = list(character_data.social_contact[4])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SKILLS_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SKILLS_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DISLIKE_TARGET_SCIENCE_INTEREST)
def handle_know_dislike_target_science_interest(character_id: int) -> int:
    """
    校验角色是否了解讨厌的人的科学知识天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(4, set())
    if len(character_data.social_contact[4]) == 0:
        return 1
    character_list = list(character_data.social_contact[4])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SCIENCE_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SCIENCE_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DISLIKE_TARGET_MYSTICS_INTEREST)
def handle_know_dislike_target_mystics_interest(character_id: int) -> int:
    """
    校验角色是否了解讨厌的人的神秘学知识天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(4, set())
    if len(character_data.social_contact[4]) == 0:
        return 1
    character_list = list(character_data.social_contact[4])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MYSTICS_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MYSTICS_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DISLIKE_TARGET_LANGUAGE_INTEREST)
def handle_know_dislike_target_language_interest(character_id: int) -> int:
    """
    校验角色是否了解讨厌的人的语言天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(4, set())
    if len(character_data.social_contact[4]) == 0:
        return 1
    character_list = list(character_data.social_contact[4])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_LANGUAGE_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LANGUAGE_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DISLIKE_TARGET_LIKE_DRESSING_STYLE)
def handle_know_dislike_target_like_dressing_style(character_id: int) -> int:
    """
    校验角色是否了解讨厌的人喜欢的穿搭风格
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(4, set())
    if len(character_data.social_contact[4]) == 0:
        return 1
    character_list = list(character_data.social_contact[4])
    target_id = random.choice(character_list)
    if constant.Awareness.KNOW_LIKE_DRESSING_STYLE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LIKE_DRESSING_STYLE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DISLIKE_TARGET_MENTAL_STATUS)
def handle_know_dislike_target_mental_status(character_id: int) -> int:
    """
    校验角色是否了解讨厌的人的心灵状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(4, set())
    if len(character_data.social_contact[4]) == 0:
        return 1
    character_list = list(character_data.social_contact[4])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MENTAL_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MENTAL_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DISLIKE_TARGET_BODILY_STATUS)
def handle_know_dislike_target_bodily_status(character_id: int) -> int:
    """
    校验角色是否了解讨厌的人的肉体状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(4, set())
    if len(character_data.social_contact[4]) == 0:
        return 1
    character_list = list(character_data.social_contact[4])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_BODILY_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_BODILY_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DISLIKE_TARGET_STATURE)
def handle_know_dislike_target_stature(character_id: int) -> int:
    """
    校验角色是否了解讨厌的人的身材
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(4, set())
    if len(character_data.social_contact[4]) == 0:
        return 1
    character_list = list(character_data.social_contact[4])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_STATURE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_ADORE_TARGET_STATURE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DISLIKE_TARGET_WEARS)
def handle_know_dislike_target_wears(character_id: int) -> int:
    """
    校验角色是否了解讨厌的人的当前的穿着
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(4, set())
    if len(character_data.social_contact[4]) == 0:
        return 1
    character_list = list(character_data.social_contact[4])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_ADORE_TARGET_WEARS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_ADORE_TARGET_WEARS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1



@handle_premise.add_premise(constant.Premise.KNOW_DISLIKE_TARGET_SEX_EXPERIENCE)
def handle_know_dislike_target_sex_experience(character_id: int) -> int:
    """
    校验角色是否了解讨厌的人的性经验
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(4, set())
    if len(character_data.social_contact[4]) == 0:
        return 1
    character_list = list(character_data.social_contact[4])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_ADORE_TARGET_SEX_EXPERIENCE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_ADORE_TARGET_SEX_EXPERIENCE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DISLIKE_TARGET_SOCIAL_STATUS)
def handle_know_dislike_target_social_status(character_id: int) -> int:
    """
    校验角色是否了解讨厌的人的社交关系
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(4, set())
    if len(character_data.social_contact[4]) == 0:
        return 1
    character_list = list(character_data.social_contact[4])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SOCIAL_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SOCIAL_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DISLIKE_TARGET_POSITION)
def handle_know_dislike_target_position(character_id: int) -> int:
    """
    校验角色是否了解讨厌的人所处的地点位置
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(4, set())
    if len(character_data.social_contact[4]) == 0:
        return 1
    character_list = list(character_data.social_contact[4])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_POSITION not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_POSITION]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DETEST_TARGET_TEMPERAMENT)
def handle_know_detest_target_temperament(character_id: int) -> int:
    """
    校验角色是否了解厌恶的人的性格
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(3, set())
    if len(character_data.social_contact[3]) == 0:
        return 0
    character_list = list(character_data.social_contact[3])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_TEMPERAMENT not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_TEMPERAMENT]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DETEST_TARGET_IDENTITY)
def handle_know_detest_target_identity(character_id: int) -> int:
    """
    校验角色是否了解厌恶的人的身份
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(3, set())
    if len(character_data.social_contact[3]) == 0:
        return 0
    character_list = list(character_data.social_contact[3])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_IDENTITY not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_IDENTITY]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DETEST_TARGET_GENDER)
def handle_know_detest_target_gender(character_id: int) -> int:
    """
    校验角色是否了解厌恶的人的性别
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(3, set())
    if len(character_data.social_contact[3]) == 0:
        return 0
    character_list = list(character_data.social_contact[3])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_GENDER not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_GENDER]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DETEST_TARGET_HUMANITY)
def handle_know_detest_target_humanity(character_id: int) -> int:
    """
    校验角色是否了解厌恶的人的人文水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(3, set())
    if len(character_data.social_contact[3]) == 0:
        return 0
    character_list = list(character_data.social_contact[3])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_HUMANITY not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_HUMANITY]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DETEST_TARGET_SKILLS)
def handle_know_detest_target_skills(character_id: int) -> int:
    """
    校验角色是否了解厌恶的人的技巧水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(3, set())
    if len(character_data.social_contact[3]) == 0:
        return 0
    character_list = list(character_data.social_contact[3])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SKILLS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SKILLS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DETEST_TARGET_SCIENCE)
def handle_know_detest_target_science(character_id: int) -> int:
    """
    校验角色是否了解厌恶的人的科学知识水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(3, set())
    if len(character_data.social_contact[3]) == 0:
        return 1
    character_list = list(character_data.social_contact[3])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SCIENCE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SCIENCE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DETEST_TARGET_MYSTICS)
def handle_know_detest_target_mystics(character_id: int) -> int:
    """
    校验角色是否了解厌恶的人的神秘学知识水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(3, set())
    if len(character_data.social_contact[3]) == 0:
        return 1
    character_list = list(character_data.social_contact[3])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MYSTICS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MYSTICS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DETEST_TARGET_LANGUAGE)
def handle_know_detest_target_language(character_id: int) -> int:
    """
    校验角色是否了解厌恶的人的语言水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(3, set())
    if len(character_data.social_contact[3]) == 0:
        return 1
    character_list = list(character_data.social_contact[3])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_LANGUAGE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LANGUAGE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DETEST_TARGET_HUMANITY_INTEREST)
def handle_know_detest_target_humanity_interest(character_id: int) -> int:
    """
    校验角色是否了解厌恶的人的人文天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(3, set())
    if len(character_data.social_contact[3]) == 0:
        return 1
    character_list = list(character_data.social_contact[3])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_HUMANITY_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_HUMANITY_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DETEST_TARGET_SKILLS_INTEREST)
def handle_know_detest_target_skills_interest(character_id: int) -> int:
    """
    校验角色是否了解厌恶的人的技巧天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(3, set())
    if len(character_data.social_contact[3]) == 0:
        return 1
    character_list = list(character_data.social_contact[3])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SKILLS_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SKILLS_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DETEST_TARGET_SCIENCE_INTEREST)
def handle_know_detest_target_science_interest(character_id: int) -> int:
    """
    校验角色是否了解厌恶的人的科学知识天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(3, set())
    if len(character_data.social_contact[3]) == 0:
        return 1
    character_list = list(character_data.social_contact[3])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SCIENCE_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SCIENCE_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DETEST_TARGET_MYSTICS_INTEREST)
def handle_know_detest_target_mystics_interest(character_id: int) -> int:
    """
    校验角色是否了解厌恶的人的神秘学知识天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(3, set())
    if len(character_data.social_contact[3]) == 0:
        return 1
    character_list = list(character_data.social_contact[3])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MYSTICS_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MYSTICS_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DETEST_TARGET_LANGUAGE_INTEREST)
def handle_know_detest_target_language_interest(character_id: int) -> int:
    """
    校验角色是否了解厌恶的人的语言天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(3, set())
    if len(character_data.social_contact[3]) == 0:
        return 1
    character_list = list(character_data.social_contact[3])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_LANGUAGE_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LANGUAGE_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DETEST_TARGET_LIKE_DRESSING_STYLE)
def handle_know_detest_target_like_dressing_style(character_id: int) -> int:
    """
    校验角色是否了解厌恶的人喜欢的穿搭风格
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(3, set())
    if len(character_data.social_contact[3]) == 0:
        return 1
    character_list = list(character_data.social_contact[3])
    target_id = random.choice(character_list)
    if constant.Awareness.KNOW_LIKE_DRESSING_STYLE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LIKE_DRESSING_STYLE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DETEST_TARGET_MENTAL_STATUS)
def handle_know_detest_target_mental_status(character_id: int) -> int:
    """
    校验角色是否了解厌恶的人的心灵状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(3, set())
    if len(character_data.social_contact[3]) == 0:
        return 1
    character_list = list(character_data.social_contact[3])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MENTAL_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MENTAL_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DETEST_TARGET_BODILY_STATUS)
def handle_know_detest_target_bodily_status(character_id: int) -> int:
    """
    校验角色是否了解厌恶的人的肉体状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(3, set())
    if len(character_data.social_contact[3]) == 0:
        return 1
    character_list = list(character_data.social_contact[3])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_BODILY_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_BODILY_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DETEST_TARGET_STATURE)
def handle_know_detest_target_stature(character_id: int) -> int:
    """
    校验角色是否了解厌恶的人的身材
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(3, set())
    if len(character_data.social_contact[3]) == 0:
        return 1
    character_list = list(character_data.social_contact[3])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_STATURE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_ADORE_TARGET_STATURE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DETEST_TARGET_WEARS)
def handle_know_detest_target_wears(character_id: int) -> int:
    """
    校验角色是否了解厌恶的人的当前的穿着
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(3, set())
    if len(character_data.social_contact[3]) == 0:
        return 1
    character_list = list(character_data.social_contact[3])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_ADORE_TARGET_WEARS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_ADORE_TARGET_WEARS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DETEST_TARGET_SEX_EXPERIENCE)
def handle_know_detest_target_sex_experience(character_id: int) -> int:
    """
    校验角色是否了解厌恶的人的性经验
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(3, set())
    if len(character_data.social_contact[3]) == 0:
        return 1
    character_list = list(character_data.social_contact[3])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_ADORE_TARGET_SEX_EXPERIENCE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_ADORE_TARGET_SEX_EXPERIENCE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DETEST_TARGET_SOCIAL_STATUS)
def handle_know_detest_target_social_status(character_id: int) -> int:
    """
    校验角色是否了解厌恶的人的社交关系
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(3, set())
    if len(character_data.social_contact[3]) == 0:
        return 1
    character_list = list(character_data.social_contact[3])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SOCIAL_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SOCIAL_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DETEST_TARGET_POSITION)
def handle_know_detest_target_position(character_id: int) -> int:
    """
    校验角色是否了解厌恶的人所处的地点位置
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(3, set())
    if len(character_data.social_contact[3]) == 0:
        return 1
    character_list = list(character_data.social_contact[3])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_POSITION not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_POSITION]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATE_TARGET_TEMPERAMENT)
def handle_know_hate_target_temperament(character_id: int) -> int:
    """
    校验角色是否了解憎恨的人的性格
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(2, set())
    if len(character_data.social_contact[2]) == 0:
        return 0
    character_list = list(character_data.social_contact[2])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_TEMPERAMENT not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_TEMPERAMENT]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATE_TARGET_IDENTITY)
def handle_know_hate_target_identity(character_id: int) -> int:
    """
    校验角色是否了解憎恨的人的身份
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(2, set())
    if len(character_data.social_contact[2]) == 0:
        return 0
    character_list = list(character_data.social_contact[2])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_IDENTITY not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_IDENTITY]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATE_TARGET_GENDER)
def handle_know_hate_target_gender(character_id: int) -> int:
    """
    校验角色是否了解憎恨的人的性别
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(2, set())
    if len(character_data.social_contact[2]) == 0:
        return 0
    character_list = list(character_data.social_contact[2])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_GENDER not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_GENDER]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATE_TARGET_HUMANITY)
def handle_know_hate_target_humanity(character_id: int) -> int:
    """
    校验角色是否了解憎恨的人的人文水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(2, set())
    if len(character_data.social_contact[2]) == 0:
        return 0
    character_list = list(character_data.social_contact[2])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_HUMANITY not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_HUMANITY]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATE_TARGET_SKILLS)
def handle_know_hate_target_skills(character_id: int) -> int:
    """
    校验角色是否了解憎恨的人的技巧水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(2, set())
    if len(character_data.social_contact[2]) == 0:
        return 0
    character_list = list(character_data.social_contact[2])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SKILLS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SKILLS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATE_TARGET_SCIENCE)
def handle_know_hate_target_science(character_id: int) -> int:
    """
    校验角色是否了解憎恨的人的科学知识水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(2, set())
    if len(character_data.social_contact[2]) == 0:
        return 1
    character_list = list(character_data.social_contact[2])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SCIENCE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SCIENCE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATE_TARGET_MYSTICS)
def handle_know_hate_target_mystics(character_id: int) -> int:
    """
    校验角色是否了解憎恨的人的神秘学知识水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(2, set())
    if len(character_data.social_contact[2]) == 0:
        return 1
    character_list = list(character_data.social_contact[2])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MYSTICS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MYSTICS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATE_TARGET_LANGUAGE)
def handle_know_hate_target_language(character_id: int) -> int:
    """
    校验角色是否了解憎恨的人的语言水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(2, set())
    if len(character_data.social_contact[2]) == 0:
        return 1
    character_list = list(character_data.social_contact[2])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_LANGUAGE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LANGUAGE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATE_TARGET_HUMANITY_INTEREST)
def handle_know_hate_target_humanity_interest(character_id: int) -> int:
    """
    校验角色是否了解憎恨的人的人文天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(2, set())
    if len(character_data.social_contact[2]) == 0:
        return 1
    character_list = list(character_data.social_contact[2])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_HUMANITY_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_HUMANITY_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATE_TARGET_SKILLS_INTEREST)
def handle_know_hate_target_skills_interest(character_id: int) -> int:
    """
    校验角色是否了解憎恨的人的技巧天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(2, set())
    if len(character_data.social_contact[2]) == 0:
        return 1
    character_list = list(character_data.social_contact[2])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SKILLS_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SKILLS_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATE_TARGET_SCIENCE_INTEREST)
def handle_know_hate_target_science_interest(character_id: int) -> int:
    """
    校验角色是否了解憎恨的人的科学知识天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(2, set())
    if len(character_data.social_contact[2]) == 0:
        return 1
    character_list = list(character_data.social_contact[2])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SCIENCE_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SCIENCE_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATE_TARGET_MYSTICS_INTEREST)
def handle_know_hate_target_mystics_interest(character_id: int) -> int:
    """
    校验角色是否了解憎恨的人的神秘学知识天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(2, set())
    if len(character_data.social_contact[2]) == 0:
        return 1
    character_list = list(character_data.social_contact[2])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MYSTICS_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MYSTICS_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATE_TARGET_LANGUAGE_INTEREST)
def handle_know_hate_target_language_interest(character_id: int) -> int:
    """
    校验角色是否了解憎恨的人的语言天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(2, set())
    if len(character_data.social_contact[2]) == 0:
        return 1
    character_list = list(character_data.social_contact[2])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_LANGUAGE_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LANGUAGE_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATE_TARGET_LIKE_DRESSING_STYLE)
def handle_know_hate_target_like_dressing_style(character_id: int) -> int:
    """
    校验角色是否了解憎恨的人喜欢的穿搭风格
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(2, set())
    if len(character_data.social_contact[2]) == 0:
        return 1
    character_list = list(character_data.social_contact[2])
    target_id = random.choice(character_list)
    if constant.Awareness.KNOW_LIKE_DRESSING_STYLE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LIKE_DRESSING_STYLE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATE_TARGET_MENTAL_STATUS)
def handle_know_hate_target_mental_status(character_id: int) -> int:
    """
    校验角色是否了解憎恨的人的心灵状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(2, set())
    if len(character_data.social_contact[2]) == 0:
        return 1
    character_list = list(character_data.social_contact[2])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MENTAL_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MENTAL_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATE_TARGET_BODILY_STATUS)
def handle_know_hate_target_bodily_status(character_id: int) -> int:
    """
    校验角色是否了解憎恨的人的肉体状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(2, set())
    if len(character_data.social_contact[2]) == 0:
        return 1
    character_list = list(character_data.social_contact[2])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_BODILY_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_BODILY_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATE_TARGET_STATURE)
def handle_know_hate_target_stature(character_id: int) -> int:
    """
    校验角色是否了解憎恨的人的身材
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(2, set())
    if len(character_data.social_contact[2]) == 0:
        return 1
    character_list = list(character_data.social_contact[2])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_STATURE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_ADORE_TARGET_STATURE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATE_TARGET_WEARS)
def handle_know_hate_target_wears(character_id: int) -> int:
    """
    校验角色是否了解憎恨的人的当前的穿着
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(2, set())
    if len(character_data.social_contact[2]) == 0:
        return 1
    character_list = list(character_data.social_contact[2])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_ADORE_TARGET_WEARS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_ADORE_TARGET_WEARS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1



@handle_premise.add_premise(constant.Premise.KNOW_HATE_TARGET_SEX_EXPERIENCE)
def handle_know_hate_target_sex_experience(character_id: int) -> int:
    """
    校验角色是否了解憎恨的人的性经验
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(2, set())
    if len(character_data.social_contact[2]) == 0:
        return 1
    character_list = list(character_data.social_contact[2])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_ADORE_TARGET_SEX_EXPERIENCE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_ADORE_TARGET_SEX_EXPERIENCE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATE_TARGET_SOCIAL_STATUS)
def handle_know_hate_target_social_status(character_id: int) -> int:
    """
    校验角色是否了解憎恨的人的社交关系
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(2, set())
    if len(character_data.social_contact[2]) == 0:
        return 1
    character_list = list(character_data.social_contact[2])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SOCIAL_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SOCIAL_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATE_TARGET_POSITION)
def handle_know_hate_target_position(character_id: int) -> int:
    """
    校验角色是否了解憎恨的人所处的地点位置
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(2, set())
    if len(character_data.social_contact[2]) == 0:
        return 1
    character_list = list(character_data.social_contact[2])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_POSITION not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_POSITION]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATRED_TARGET_TEMPERAMENT)
def handle_know_hatred_target_temperament(character_id: int) -> int:
    """
    校验角色是否了解仇视的人的性格
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(1, set())
    if len(character_data.social_contact[1]) == 0:
        return 0
    character_list = list(character_data.social_contact[1])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_TEMPERAMENT not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_TEMPERAMENT]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATRED_TARGET_IDENTITY)
def handle_know_hatred_target_identity(character_id: int) -> int:
    """
    校验角色是否了解仇视的人的身份
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(1, set())
    if len(character_data.social_contact[1]) == 0:
        return 0
    character_list = list(character_data.social_contact[1])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_IDENTITY not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_IDENTITY]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATRED_TARGET_GENDER)
def handle_know_hatred_target_gender(character_id: int) -> int:
    """
    校验角色是否了解仇视的人的性别
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(1, set())
    if len(character_data.social_contact[1]) == 0:
        return 0
    character_list = list(character_data.social_contact[1])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_GENDER not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_GENDER]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATRED_TARGET_HUMANITY)
def handle_know_hatred_target_humanity(character_id: int) -> int:
    """
    校验角色是否了解仇视的人的人文水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(1, set())
    if len(character_data.social_contact[1]) == 0:
        return 0
    character_list = list(character_data.social_contact[1])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_HUMANITY not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_HUMANITY]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATRED_TARGET_SKILLS)
def handle_know_hatred_target_skills(character_id: int) -> int:
    """
    校验角色是否了解仇视的人的技巧水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(1, set())
    if len(character_data.social_contact[1]) == 0:
        return 0
    character_list = list(character_data.social_contact[1])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SKILLS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SKILLS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATRED_TARGET_SCIENCE)
def handle_know_hatred_target_science(character_id: int) -> int:
    """
    校验角色是否了解仇视的人的科学知识水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(1, set())
    if len(character_data.social_contact[1]) == 0:
        return 1
    character_list = list(character_data.social_contact[1])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SCIENCE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SCIENCE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATRED_TARGET_MYSTICS)
def handle_know_hatred_target_mystics(character_id: int) -> int:
    """
    校验角色是否了解仇视的人的神秘学知识水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(1, set())
    if len(character_data.social_contact[1]) == 0:
        return 1
    character_list = list(character_data.social_contact[1])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MYSTICS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MYSTICS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATRED_TARGET_LANGUAGE)
def handle_know_hatred_target_language(character_id: int) -> int:
    """
    校验角色是否了解仇视的人的语言水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(1, set())
    if len(character_data.social_contact[1]) == 0:
        return 1
    character_list = list(character_data.social_contact[1])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_LANGUAGE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LANGUAGE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATRED_TARGET_HUMANITY_INTEREST)
def handle_know_hatred_target_humanity_interest(character_id: int) -> int:
    """
    校验角色是否了解仇视的人的人文天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(1, set())
    if len(character_data.social_contact[1]) == 0:
        return 1
    character_list = list(character_data.social_contact[1])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_HUMANITY_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_HUMANITY_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATRED_TARGET_SKILLS_INTEREST)
def handle_know_hatred_target_skills_interest(character_id: int) -> int:
    """
    校验角色是否了解仇视的人的技巧天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(1, set())
    if len(character_data.social_contact[1]) == 0:
        return 1
    character_list = list(character_data.social_contact[1])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SKILLS_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SKILLS_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATRED_TARGET_SCIENCE_INTEREST)
def handle_know_hatred_target_science_interest(character_id: int) -> int:
    """
    校验角色是否了解仇视的人的科学知识天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(1, set())
    if len(character_data.social_contact[1]) == 0:
        return 1
    character_list = list(character_data.social_contact[1])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SCIENCE_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SCIENCE_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATRED_TARGET_MYSTICS_INTEREST)
def handle_know_hatred_target_mystics_interest(character_id: int) -> int:
    """
    校验角色是否了解仇视的人的神秘学知识天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(1, set())
    if len(character_data.social_contact[1]) == 0:
        return 1
    character_list = list(character_data.social_contact[1])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MYSTICS_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MYSTICS_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATRED_TARGET_LANGUAGE_INTEREST)
def handle_know_hatred_target_language_interest(character_id: int) -> int:
    """
    校验角色是否了解仇视的人的语言天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(1, set())
    if len(character_data.social_contact[1]) == 0:
        return 1
    character_list = list(character_data.social_contact[1])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_LANGUAGE_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LANGUAGE_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATRED_TARGET_LIKE_DRESSING_STYLE)
def handle_know_hatred_target_like_dressing_style(character_id: int) -> int:
    """
    校验角色是否了解仇视的人喜欢的穿搭风格
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(1, set())
    if len(character_data.social_contact[1]) == 0:
        return 1
    character_list = list(character_data.social_contact[1])
    target_id = random.choice(character_list)
    if constant.Awareness.KNOW_LIKE_DRESSING_STYLE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LIKE_DRESSING_STYLE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATRED_TARGET_MENTAL_STATUS)
def handle_know_hatred_target_mental_status(character_id: int) -> int:
    """
    校验角色是否了解仇视的人的心灵状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(1, set())
    if len(character_data.social_contact[1]) == 0:
        return 1
    character_list = list(character_data.social_contact[1])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MENTAL_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MENTAL_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATRED_TARGET_BODILY_STATUS)
def handle_know_hatred_target_bodily_status(character_id: int) -> int:
    """
    校验角色是否了解仇视的人的肉体状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(1, set())
    if len(character_data.social_contact[1]) == 0:
        return 1
    character_list = list(character_data.social_contact[1])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_BODILY_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_BODILY_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATRED_TARGET_STATURE)
def handle_know_hatred_target_stature(character_id: int) -> int:
    """
    校验角色是否了解仇视的人的身材
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(1, set())
    if len(character_data.social_contact[1]) == 0:
        return 1
    character_list = list(character_data.social_contact[1])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_STATURE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_ADORE_TARGET_STATURE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATRED_TARGET_WEARS)
def handle_know_hatred_target_wears(character_id: int) -> int:
    """
    校验角色是否了解仇视的人的当前的穿着
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(1, set())
    if len(character_data.social_contact[1]) == 0:
        return 1
    character_list = list(character_data.social_contact[1])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_ADORE_TARGET_WEARS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_ADORE_TARGET_WEARS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATRED_TARGET_SEX_EXPERIENCE)
def handle_know_hatred_target_sex_experience(character_id: int) -> int:
    """
    校验角色是否了解仇视的人的性经验
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(1, set())
    if len(character_data.social_contact[1]) == 0:
        return 1
    character_list = list(character_data.social_contact[1])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_ADORE_TARGET_SEX_EXPERIENCE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_ADORE_TARGET_SEX_EXPERIENCE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATRED_TARGET_SOCIAL_STATUS)
def handle_know_hatred_target_social_status(character_id: int) -> int:
    """
    校验角色是否了解仇视的人的社交关系
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(1, set())
    if len(character_data.social_contact[1]) == 0:
        return 1
    character_list = list(character_data.social_contact[1])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SOCIAL_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SOCIAL_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_HATRED_TARGET_POSITION)
def handle_know_hatred_target_position(character_id: int) -> int:
    """
    校验角色是否了解仇视的人所处的地点位置
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(1, set())
    if len(character_data.social_contact[1]) == 0:
        return 1
    character_list = list(character_data.social_contact[1])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_POSITION not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_POSITION]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEADLY_ENEMY_TEMPERAMENT)
def handle_know_deadly_enemy_temperament(character_id: int) -> int:
    """
    校验角色是否了解死敌的性格
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(0, set())
    if len(character_data.social_contact[0]) == 0:
        return 0
    character_list = list(character_data.social_contact[0])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_TEMPERAMENT not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_TEMPERAMENT]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEADLY_ENEMY_IDENTITY)
def handle_know_deadly_enemy_identity(character_id: int) -> int:
    """
    校验角色是否了解死敌的身份
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(0, set())
    if len(character_data.social_contact[0]) == 0:
        return 0
    character_list = list(character_data.social_contact[0])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_IDENTITY not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_IDENTITY]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEADLY_ENEMY_GENDER)
def handle_know_deadly_enemy_gender(character_id: int) -> int:
    """
    校验角色是否了解死敌的性别
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(0, set())
    if len(character_data.social_contact[0]) == 0:
        return 0
    character_list = list(character_data.social_contact[0])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_GENDER not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_GENDER]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEADLY_ENEMY_HUMANITY)
def handle_know_deadly_enemy_humanity(character_id: int) -> int:
    """
    校验角色是否了解死敌的人文水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(0, set())
    if len(character_data.social_contact[0]) == 0:
        return 0
    character_list = list(character_data.social_contact[0])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_HUMANITY not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_HUMANITY]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEADLY_ENEMY_SKILLS)
def handle_know_deadly_enemy_skills(character_id: int) -> int:
    """
    校验角色是否了解死敌的技巧水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(0, set())
    if len(character_data.social_contact[0]) == 0:
        return 0
    character_list = list(character_data.social_contact[0])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SKILLS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SKILLS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEADLY_ENEMY_SCIENCE)
def handle_know_deadly_enemy_science(character_id: int) -> int:
    """
    校验角色是否了解死敌的科学知识水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(0, set())
    if len(character_data.social_contact[0]) == 0:
        return 1
    character_list = list(character_data.social_contact[0])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SCIENCE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SCIENCE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEADLY_ENEMY_MYSTICS)
def handle_know_deadly_enemy_mystics(character_id: int) -> int:
    """
    校验角色是否了解死敌的神秘学知识水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(0, set())
    if len(character_data.social_contact[0]) == 0:
        return 1
    character_list = list(character_data.social_contact[0])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MYSTICS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MYSTICS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEADLY_ENEMY_LANGUAGE)
def handle_know_deadly_enemy_language(character_id: int) -> int:
    """
    校验角色是否了解死敌的语言水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(0, set())
    if len(character_data.social_contact[0]) == 0:
        return 1
    character_list = list(character_data.social_contact[0])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_LANGUAGE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LANGUAGE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEADLY_ENEMY_HUMANITY_INTEREST)
def handle_know_deadly_enemy_humanity_interest(character_id: int) -> int:
    """
    校验角色是否了解死敌的人文天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(0, set())
    if len(character_data.social_contact[0]) == 0:
        return 1
    character_list = list(character_data.social_contact[0])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_HUMANITY_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_HUMANITY_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEADLY_ENEMY_SKILLS_INTEREST)
def handle_know_deadly_enemy_skills_interest(character_id: int) -> int:
    """
    校验角色是否了解死敌的技巧天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(0, set())
    if len(character_data.social_contact[0]) == 0:
        return 1
    character_list = list(character_data.social_contact[0])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SKILLS_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SKILLS_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEADLY_ENEMY_SCIENCE_INTEREST)
def handle_know_deadly_enemy_science_interest(character_id: int) -> int:
    """
    校验角色是否了解死敌的科学知识天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(0, set())
    if len(character_data.social_contact[0]) == 0:
        return 1
    character_list = list(character_data.social_contact[0])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SCIENCE_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SCIENCE_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEADLY_ENEMY_MYSTICS_INTEREST)
def handle_know_deadly_enemy_mystics_interest(character_id: int) -> int:
    """
    校验角色是否了解死敌的神秘学知识天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(0, set())
    if len(character_data.social_contact[0]) == 0:
        return 1
    character_list = list(character_data.social_contact[0])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MYSTICS_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MYSTICS_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEADLY_ENEMY_LANGUAGE_INTEREST)
def handle_know_deadly_enemy_language_interest(character_id: int) -> int:
    """
    校验角色是否了解死敌的语言天赋水平
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(0, set())
    if len(character_data.social_contact[0]) == 0:
        return 1
    character_list = list(character_data.social_contact[0])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_LANGUAGE_INTEREST not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LANGUAGE_INTEREST]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEADLY_ENEMY_LIKE_DRESSING_STYLE)
def handle_know_deadly_enemy_like_dressing_style(character_id: int) -> int:
    """
    校验角色是否了解死敌喜欢的穿搭风格
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(0, set())
    if len(character_data.social_contact[0]) == 0:
        return 1
    character_list = list(character_data.social_contact[0])
    target_id = random.choice(character_list)
    if constant.Awareness.KNOW_LIKE_DRESSING_STYLE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_LIKE_DRESSING_STYLE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEADLY_ENEMY_MENTAL_STATUS)
def handle_know_deadly_enemy_mental_status(character_id: int) -> int:
    """
    校验角色是否了解死敌的心灵状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(0, set())
    if len(character_data.social_contact[0]) == 0:
        return 1
    character_list = list(character_data.social_contact[0])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_MENTAL_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_MENTAL_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEADLY_ENEMY_BODILY_STATUS)
def handle_know_deadly_enemy_bodily_status(character_id: int) -> int:
    """
    校验角色是否了解死敌的肉体状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(0, set())
    if len(character_data.social_contact[0]) == 0:
        return 1
    character_list = list(character_data.social_contact[0])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_BODILY_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_BODILY_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEADLY_ENEMY_STATURE)
def handle_know_deadly_enemy_stature(character_id: int) -> int:
    """
    校验角色是否了解死敌的身材
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(0, set())
    if len(character_data.social_contact[0]) == 0:
        return 1
    character_list = list(character_data.social_contact[0])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_STATURE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_ADORE_TARGET_STATURE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEADLY_ENEMY_WEARS)
def handle_know_deadly_enemy_wears(character_id: int) -> int:
    """
    校验角色是否了解死敌的当前的穿着
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(0, set())
    if len(character_data.social_contact[0]) == 0:
        return 1
    character_list = list(character_data.social_contact[0])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_ADORE_TARGET_WEARS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_ADORE_TARGET_WEARS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEADLY_ENEMY_SEX_EXPERIENCE)
def handle_know_deadly_enemy_sex_experience(character_id: int) -> int:
    """
    校验角色是否了解死敌的性经验
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(0, set())
    if len(character_data.social_contact[0]) == 0:
        return 1
    character_list = list(character_data.social_contact[0])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_ADORE_TARGET_SEX_EXPERIENCE not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_ADORE_TARGET_SEX_EXPERIENCE]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEADLY_ENEMY_SOCIAL_STATUS)
def handle_know_deadly_enemy_social_status(character_id: int) -> int:
    """
    校验角色是否了解死敌的社交关系
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(0, set())
    if len(character_data.social_contact[0]) == 0:
        return 1
    character_list = list(character_data.social_contact[0])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_SOCIAL_STATUS not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_SOCIAL_STATUS]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.KNOW_DEADLY_ENEMY_POSITION)
def handle_know_deadly_enemy_position(character_id: int) -> int:
    """
    校验角色是否了解死敌所处的地点位置
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(0, set())
    if len(character_data.social_contact[0]) == 0:
        return 1
    character_list = list(character_data.social_contact[0])
    target_id = random.choice(character_list)
    if target_id not in character_data.character_awareness_data:
        return 0
    if constant.Awareness.KNOW_POSITION not in character_data.character_awareness_data[target_id]:
        return 0
    awareness_data = character_data.character_awareness_data[target_id][constant.Awareness.KNOW_POSITION]
    confidence_probability = awareness_data.got_count * 10
    check_probability = random.randint(0, 100)
    if check_probability > confidence_probability:
        return 0
    if not awareness_data.authenticity:
        return 0
    return 1
