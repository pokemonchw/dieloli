from Script.Design import handle_premise, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.TARGET_IS_ADORE)
def handle_target_is_adore(character_id: int) -> int:
    """
    校验角色当前目标是否是自己的爱慕对象
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    target_id = character_data.target_character_id
    character_data.social_contact.setdefault(10, set())
    if target_id in character_data.social_contact[10]:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_IS_ADMIRE)
def handle_target_is_admire(character_id: int) -> int:
    """
    校验角色当前的目标是否是自己的恋慕对象
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    target_id = character_data.target_character_id
    character_data.social_contact.setdefault(9, set())
    if target_id in character_data.social_contact[9]:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.PLAYER_IS_ADORE)
def handle_player_is_adore(character_id: int) -> int:
    """
    校验玩家是否是当前角色的爱慕对象
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    character_data.social_contact.setdefault(10, set())
    if 0 in character_data.social_contact[10]:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_IS_BEYOND_FRIENDSHIP)
def handle_target_is_beyond_friendship(character_id: int) -> int:
    """
    校验是否对目标抱有超越友谊的想法
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if (
        character_data.target_character_id in character_data.social_contact_data
        and character_data.social_contact_data[character_data.target_character_id] > 7
    ):
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_BEYOND_FRIENDSHIP_TARGET)
def handle_is_beyond_friendship_target(character_id: int) -> int:
    """
    校验目标是否对自己抱有超越友谊的想法
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if (
        character_id in target_data.social_contact_data
        and target_data.social_contact_data[character_id] > 7
    ):
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_ADORE)
def handle_target_adore(character_id: int) -> int:
    """
    校验是否被交互对象爱慕
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.social_contact.setdefault(10, set())
    if character_id in target_data.social_contact[10]:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.NO_BEYOND_FRIENDSHIP_TARGET)
def handle_no_beyond_friendship_target(character_id: int) -> int:
    """
    校验目标是否对自己没有超越友谊的想法
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if (
        character_id in target_data.social_contact_data
        and target_data.social_contact_data[character_id] < 8
    ):
        return 1
    if character_id not in target_data.social_contact_data:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_ADMIRE)
def handle_target_admire(character_id: int) -> int:
    """
    校验角色是否被交互对象恋慕
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.social_contact.setdefault(9, set())
    if character_id in target_data.social_contact[9]:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.HAVE_LIKE_TARGET)
def handle_have_like_target(character_id: int) -> int:
    """
    校验是否有喜欢的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(9, set())
    character_data.social_contact.setdefault(10, set())
    return (len(character_data.social_contact[9]) + len(character_data.social_contact[10])) > 0


@handle_premise.add_premise(constant.Premise.HAVE_DISLIKE_TARGET)
def handle_have_dislike_target(character_id: int) -> int:
    """
    校验是否有讨厌的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(0, set())
    character_data.social_contact.setdefault(1, set())
    character_data.social_contact.setdefault(2, set())
    character_data.social_contact.setdefault(3, set())
    return (len(character_data.social_contact[0]) + len(character_data.social_contact[1]) + len(character_data.social_contact[2]) + len(character_data.social_contact[3])) > 0


@handle_premise.add_premise(constant.Premise.TARGET_NOT_STRANGER)
def handle_target_not_stranger(character_id: int) -> int:
    """
    校验交互对象是否不是陌生人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id in character_data.social_contact_data:
        target_data: game_type.Character = cache.character_data[character_data.target_character_id]
        if character_id in target_data.social_contact_data:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.DISLIKE_TARGET)
def handle_dislike_target(character_id: int) -> int:
    """
    校验是否讨厌交互对象
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id in character_data.social_contact_data:
        social_type = character_data.social_contact_data[character_data.target_character_id]
        return social_type in {0, 1, 2, 3, 4}
    return 0


@handle_premise.add_premise(constant.Premise.DETEST_TARGET)
def handle_detest_target(character_id: int) -> int:
    """
    校验是否厌恶交互对象
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id in character_data.social_contact_data:
        social_type = character_data.social_contact_data[character_data.target_character_id]
        return social_type in {0, 1, 2, 3}
    return 0


@handle_premise.add_premise(constant.Premise.HATE_TARGET)
def handle_hate_target(character_id: int) -> int:
    """
    校验是否憎恨交互对象
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id in character_data.social_contact_data:
        social_type = character_data.social_contact_data[character_data.target_character_id]
        return social_type in {0, 1, 2}
    return 0


@handle_premise.add_premise(constant.Premise.HATRED_TARGET)
def handle_hatred_target(character_id: int) -> int:
    """
    校验是否仇视交互对象
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id in character_data.social_contact_data:
        social_type = character_data.social_contact_data[character_data.target_character_id]
        return social_type in {0, 1}
    return 0


@handle_premise.add_premise(constant.Premise.DEADLY_ENEMY_TARGET)
def handle_deadly_enemy_target(character_id: int) -> int:
    """
    校验是否将交互对象视为死敌
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id in character_data.social_contact_data:
        social_type = character_data.social_contact_data[character_data.target_character_id]
        return not social_type
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_DISLIKE)
def handle_target_dislike(character_id: int) -> int:
    """
    校验是否被交互对象讨厌
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if character_id in target_data.social_contact_data:
        social_type = target_data.social_contact_data[character_id]
        return social_type in {0, 1, 2, 3, 4}
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_DETEST)
def handle_target_detest(character_id: int) -> int:
    """
    校验是否被交互对象厌恶
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if character_id in target_data.social_contact_data:
        social_type = target_data.social_contact_data[character_id]
        return social_type in {0, 1, 2, 3}
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_HATE)
def handle_target_hate(character_id: int) -> int:
    """
    校验是否被交互对象憎恨
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if character_id in target_data.social_contact_data:
        social_type = target_data.social_contact_data[character_id]
        return social_type in {0, 1, 2}
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_HATRED)
def handle_target_hatred(character_id: int) -> int:
    """
    校验是否被交互对象仇视
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if character_id in target_data.social_contact_data:
        social_type = target_data.social_contact_data[character_id]
        return social_type in {0, 1}
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_DEADLY_ENEMY)
def handle_target_deadly_enemy(character_id: int) -> int:
    """
    校验是否被交互对象视为死敌
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if character_id in target_data.social_contact_data:
        social_type = target_data.social_contact_data[character_id]
        return not social_type
    return 0
