from Script.Design import handle_premise
from Script.Core import constant, game_type, cache_control

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
    character_data.social_contact.setdefault(5, set())
    if target_id in character_data.social_contact[5]:
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
    character_data.social_contact.setdefault(4, set())
    if target_id in character_data.social_contact[4]:
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
    character_data.social_contact.setdefault(5, set())
    if 0 in character_data.social_contact[5]:
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
        and character_data.social_contact_data[character_data.target_character_id] > 2
    ):
        return character_data.social_contact_data[character_data.target_character_id]
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
        and target_data.social_contact_data[character_id] > 2
    ):
        return target_data.social_contact_data[character_id]
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
    target_data.social_contact.setdefault(5, set())
    if character_id in target_data.social_contact[5]:
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
        and target_data.social_contact_data[character_id] < 3
    ):
        return 5 - target_data.social_contact_data[character_id]
    if character_id not in target_data.social_contact_data:
        return 5
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
    target_data.social_contact.setdefault(4, set())
    if character_id in target_data.social_contact[4]:
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
    character_data.social_contact.setdefault(4, set())
    character_data.social_contact.setdefault(5, set())
    return len(character_data.social_contact[4]) + len(character_data.social_contact[5])


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
