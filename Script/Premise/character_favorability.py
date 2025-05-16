from Script.Design import handle_premise, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.TARGET_IS_ADORE)
def handle_target_is_adore(character_id: int) -> int:
    """
    校验角色是否爱慕交互对象
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id == -1:
        return 0
    if target_id in character_data.social_contact_data:
        if character_data.social_contact_data[target_id] == 10:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_IS_ADMIRE)
def handle_target_is_admire(character_id: int) -> int:
    """
    校验角色是否恋慕交互对象
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id == -1:
        return 0
    if target_id in character_data.social_contact_data:
        if character_data.social_contact_data[target_id] >= 9:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_IS_DEPEND_ON)
def handle_target_is_depend_on(character_id: int) -> int:
    """
    校验角色是否依靠交互对象
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id == -1:
        return 0
    if target_id in character_data.social_contact_data:
        if character_data.social_contact_data[target_id] >= 8:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_IS_TRUST)
def handle_target_is_trust(character_id: int) -> int:
    """
    校验角色是否信任交互对象
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id == -1:
        return 0
    if target_id in character_data.social_contact_data:
        if character_data.social_contact_data[target_id] >= 7:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_IS_FRIEND)
def handle_target_is_friend(character_id: int) -> int:
    """
    校验角色是否认为交互对象是朋友
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id == -1:
        return 0
    if target_id in character_data.social_contact_data:
        if character_data.social_contact_data[target_id] >= 6:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_IS_STRANGER)
def handle_target_is_stranger(character_id: int) -> int:
    """
    校验角色是否认为交互对象是陌生人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id == -1:
        return 0
    if target_id in character_data.social_contact_data:
        if character_data.social_contact_data[target_id] == 5:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.DISLIKE_TARGET)
def handle_dislike_target(character_id: int) -> int:
    """
    校验角色是否讨厌交互对象
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id == -1:
        return 0
    if target_id in character_data.social_contact_data:
        if character_data.social_contact_data[target_id] <= 4:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.DETEST_TARGET)
def handle_detest_target(character_id: int) -> int:
    """
    校验角色是否厌恶交互对象
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id == -1:
        return 0
    if target_id in character_data.social_contact_data:
        if character_data.social_contact_data[target_id] <= 3:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.HATE_TARGET)
def handle_hate_target(character_id: int) -> int:
    """
    校验角色是否憎恨交互对象
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if target_id == -1:
        return 0
    if target_id in character_data.social_contact_data:
        if character_data.social_contact_data[target_id] <= 2:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.HATRED_TARGET)
def handle_hatred_target(character_id: int) -> int:
    """
    校验角色是否仇视交互对象
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if target_id == -1:
        return 0
    if target_id in character_data.social_contact_data:
        if character_data.social_contact_data[target_id] <= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.DEADLY_ENEMY_TARGET)
def handle_deadly_enemy_target(character_id: int) -> int:
    """
    校验角色是否将交互对象视为死敌
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if target_id == -1:
        return 0
    if target_id in character_data.social_contact_data:
        if character_data.social_contact_data[target_id] == 0:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_ADORE)
def handle_target_adore(character_id: int) -> int:
    """
    校验角色是否被交互对象爱慕
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if character_id in target_data.social_contact[10]:
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
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if character_id in target_data.social_contact_data:
        if target_data.social_contact_data[character_id] >= 9:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_DEPEND_ON)
def handle_target_depend_on(character_id: int) -> int:
    """
    校验角色是否被交互对象依靠
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if character_id in target_data.social_contact_data:
        if target_data.social_contact_data[character_id] >= 8:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_TRUST)
def handle_target_trust(character_id: int) -> int:
    """
    校验角色是否被交互对象信任
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if character_id in target_data.social_contact_data:
        if target_data.social_contact_data[character_id] >= 7:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_FRIEND)
def handle_target_friend(character_id: int) -> int:
    """
    校验角色是否被交互对象视为朋友
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if character_id in target_data.social_contact_data:
        if target_data.social_contact_data[character_id] >= 6:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_STRANGER)
def handle_target_stranger(character_id: int) -> int:
    """
    校验角色是否被交互对象视为陌生人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if character_id in target_data.social_contact_data:
        if target_data.social_contact_data[character_id] == 5:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_DISLIKE)
def handle_target_dislike(character_id: int) -> int:
    """
    校验角色是否被交互对象讨厌
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if character_id in target_data.social_contact_data:
        if target_data.social_contact_data[character_id] <= 4:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_DETEST)
def handle_target_detest(character_id: int) -> int:
    """
    校验角色是否被交互对象厌恶
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if character_id in target_data.social_contact_data:
        if target_data.social_contact_data[character_id] <= 3:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_HATE)
def handle_target_hate(character_id: int) -> int:
    """
    校验角色是否被交互对象憎恨
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if character_id in target_data.social_contact_data:
        if target_data.social_contact_data[character_id] <= 2:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_HATRED)
def handle_target_hatred(character_id: int) -> int:
    """
    校验角色是否被交互对象仇视
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if character_id in target_data.social_contact_data:
        if target_data.social_contact_data[character_id] <= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_DEADLY_ENEMY)
def handle_target_deadly_enemy(character_id: int) -> int:
    """
    校验角色是否被交互对象视为死敌
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if character_id in target_data.social_contact_data:
        if target_data.social_contact_data[character_id] == 0:
            return 1
    return 0


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
    if character_data.target_character_id == -1:
        return 0
    if character_data.target_character_id in character_data.social_contact_data:
        target_data: game_type.Character = cache.character_data[character_data.target_character_id]
        if character_id in target_data.social_contact_data:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_LESS_THAN_ADORE)
def handle_target_less_than_adore(character_id: int) -> int:
    """
    校验交互对象对角色的好感是否小于爱慕
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if character_id not in target_data.social_contact_data:
        return 1
    if target_data.social_contact_data[character_id] < 10:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_LESS_THAN_ADMIRE)
def handle_target_less_than_admire(character_id: int) -> int:
    """
    校验交互对象对角色的好感是否小于恋慕
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if character_id not in target_data.social_contact_data:
        return 1
    if target_data.social_contact_data[character_id] < 9:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_LESS_THAN_DEPEND_ON)
def handle_target_less_than_depend_on(character_id: int) -> int:
    """
    校验交互对象对角色的好感是否小于依靠
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if character_id not in target_data.social_contact_data:
        return 1
    if target_data.social_contact_data[character_id] < 8:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_LESS_THAN_TRUST)
def handle_target_less_than_trust(character_id: int) -> int:
    """
    校验交互对象对角色的好感是否小于信任
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if character_id not in target_data.social_contact_data:
        return 1
    if target_data.social_contact_data[character_id] < 7:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_LESS_THAN_FRIEND)
def handle_target_less_than_friend(character_id: int) -> int:
    """
    校验角色是否不被交互对象视为朋友
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if character_id not in target_data.social_contact_data:
        return 1
    if target_data.social_contact_data[character_id] < 6:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_MORE_THAN_DISLIKE)
def handle_target_more_than_dislike(character_id: int) -> int:
    """
    校验角色是否不被交互对象讨厌
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if character_id not in target_data.social_contact_data:
        return 1
    if target_data.social_contact_data[character_id] > 4:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_MORE_THAN_DETEST)
def handle_target_more_than_detest(character_id: int) -> int:
    """
    校验角色是否不被交互对象厌恶
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if character_id not in target_data.social_contact_data:
        return 1
    if target_data.social_contact_data[character_id] > 3:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_MORE_THAN_HATE)
def handle_target_more_than_hate(character_id: int) -> int:
    """
    校验角色是否不被交互对象憎恨
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if character_id not in target_data.social_contact_data:
        return 1
    if target_data.social_contact_data[character_id] > 2:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_MORE_THAN_HATRED)
def handle_target_more_than_hatred(character_id: int) -> int:
    """
    校验角色是否不被交互对象仇视
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if character_id not in target_data.social_contact_data:
        return 1
    if target_data.social_contact_data[character_id] > 1:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_MORE_THAN_DEADLY_ENEMY)
def handle_target_more_than_deadly_enemy(character_id: int) -> int:
    """
    校验角色是否不被交互对象视为死敌
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if character_id not in target_data.social_contact_data:
        return 1
    if target_data.social_contact_data[character_id] > 0:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_IS_LESS_THAN_ADORE)
def handle_target_is_less_than_adore(character_id: int) -> int:
    """
    校验角色对交互对象的好感是否小于爱慕
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_id = character_data.target_character_id
    if target_id in character_data.social_contact[10]:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_IS_LESS_THAN_ADMIRE)
def handle_target_is_less_than_admire(character_id: int) -> int:
    """
    校验角色对交互对象的好感是否小于恋慕
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_id = character_data.target_character_id
    if target_id not in character_data.social_contact_data:
        return 1
    if character_data.social_contact_data[target_id] < 9:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_IS_LESS_THAN_DEPEND_ON)
def handle_target_is_less_than_depend_on(character_id: int) -> int:
    """
    校验角色对交互对象的好感是否小于依靠
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_id = character_data.target_character_id
    if target_id not in character_data.social_contact_data:
        return 1
    if character_data.social_contact_data[target_id] < 8:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_IS_LESS_THAN_TRUST)
def handle_target_is_less_than_trust(character_id: int) -> int:
    """
    校验角色对交互对象的好感是否小于信任
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_id = character_data.target_character_id
    if target_id not in character_data.social_contact_data:
        return 1
    if character_data.social_contact_data[target_id] < 7:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_IS_LESS_THAN_FRIEND)
def handle_target_is_less_than_friend(character_id: int) -> int:
    """
    校验角色是否不将交互对象视为朋友
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_id = character_data.target_character_id
    if target_id not in character_data.social_contact_data:
        return 1
    if character_data.social_contact_data[target_id] < 6:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_IS_MORE_THAN_DISLIKE)
def handle_target_is_more_than_dislike(character_id: int) -> int:
    """
    校验角色是否不讨厌交互对象
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_id = character_data.target_character_id
    if target_id not in character_data.social_contact_data:
        return 1
    if character_data.social_contact_data[target_id] > 4:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_IS_MORE_THAN_DETEST)
def handle_target_is_more_than_detest(character_id: int) -> int:
    """
    校验角色是否不厌恶交互对象
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_id = character_data.target_character_id
    if target_id not in character_data.social_contact_data:
        return 1
    if character_data.social_contact_data[target_id] > 3:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_IS_MORE_THAN_HATE)
def handle_target_is_more_than_hate(character_id: int) -> int:
    """
    校验角色是否不憎恨交互对象
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_id = character_data.target_character_id
    if target_id not in character_data.social_contact_data:
        return 1
    if character_data.social_contact_data[target_id] > 2:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_IS_MORE_THAN_HATRED)
def handle_target_is_more_than_hatred(character_id: int) -> int:
    """
    校验角色是否不仇视交互对象
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_id = character_data.target_character_id
    if target_id not in character_data.social_contact_data:
        return 1
    if character_data.social_contact_data[target_id] > 1:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_IS_MORE_THAN_DEADLY_ENEMY)
def handle_target_is_more_than_deadly_enemy(character_id: int) -> int:
    """
    校验角色是否不将交互对象视为死敌
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_id = character_data.target_character_id
    if target_id not in character_data.social_contact_data:
        return 1
    if character_data.social_contact_data[target_id] != 10:
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
    校验是否对交互对象抱有超越友谊的想法
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    if (
        character_data.target_character_id in character_data.social_contact_data
        and character_data.social_contact_data[character_data.target_character_id] > 7
    ):
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_BEYOND_FRIENDSHIP_TARGET)
def handle_is_beyond_friendship_target(character_id: int) -> int:
    """
    校验交互对象是否对自己抱有超越友谊的想法
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if (
        character_id in target_data.social_contact_data
        and target_data.social_contact_data[character_id] > 7
    ):
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.NO_BEYOND_FRIENDSHIP_TARGET)
def handle_no_beyond_friendship_target(character_id: int) -> int:
    """
    校验交互对象是否对自己没有超越友谊的想法
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 1
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if (
        character_id in target_data.social_contact_data
        and target_data.social_contact_data[character_id] < 8
    ):
        return 1
    if character_id not in target_data.social_contact_data:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.HAVE_LIKE_TARGET)
def handle_have_like_target(character_id: int) -> int:
    """
    校验角色是否有喜欢的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(9, set())
    character_data.social_contact.setdefault(10, set())
    if (len(character_data.social_contact[9]) + len(character_data.social_contact[10])) > 0:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.HAVE_DISLIKE_TARGET)


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
    校验是否对交互对象抱有超越友谊的想法
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    if (
        character_data.target_character_id in character_data.social_contact_data
        and character_data.social_contact_data[character_data.target_character_id] > 7
    ):
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_BEYOND_FRIENDSHIP_TARGET)
def handle_is_beyond_friendship_target(character_id: int) -> int:
    """
    校验交互对象是否对自己抱有超越友谊的想法
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if (
        character_id in target_data.social_contact_data
        and target_data.social_contact_data[character_id] > 7
    ):
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.NO_BEYOND_FRIENDSHIP_TARGET)
def handle_no_beyond_friendship_target(character_id: int) -> int:
    """
    校验交互对象是否对自己没有超越友谊的想法
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 1
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if (
        character_id in target_data.social_contact_data
        and target_data.social_contact_data[character_id] < 8
    ):
        return 1
    if character_id not in target_data.social_contact_data:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.HAVE_LIKE_TARGET)
def handle_have_like_target(character_id: int) -> int:
    """
    校验角色是否有喜欢的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(9, set())
    character_data.social_contact.setdefault(10, set())
    if (len(character_data.social_contact[9]) + len(character_data.social_contact[10])) > 0:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.HAVE_DISLIKE_TARGET)


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
    校验是否对交互对象抱有超越友谊的想法
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    if (
        character_data.target_character_id in character_data.social_contact_data
        and character_data.social_contact_data[character_data.target_character_id] > 7
    ):
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_BEYOND_FRIENDSHIP_TARGET)
def handle_is_beyond_friendship_target(character_id: int) -> int:
    """
    校验交互对象是否对自己抱有超越友谊的想法
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if (
        character_id in target_data.social_contact_data
        and target_data.social_contact_data[character_id] > 7
    ):
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.NO_BEYOND_FRIENDSHIP_TARGET)
def handle_no_beyond_friendship_target(character_id: int) -> int:
    """
    校验交互对象是否对自己没有超越友谊的想法
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 1
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if (
        character_id in target_data.social_contact_data
        and target_data.social_contact_data[character_id] < 8
    ):
        return 1
    if character_id not in target_data.social_contact_data:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.HAVE_LIKE_TARGET)
def handle_have_like_target(character_id: int) -> int:
    """
    校验角色是否有喜欢的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.social_contact.setdefault(9, set())
    character_data.social_contact.setdefault(10, set())
    if (len(character_data.social_contact[9]) + len(character_data.social_contact[10])) > 0:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.HAVE_DISLIKE_TARGET)
def handle_have_dislike_target(character_id: int) -> int:
    """
    校验角色是否有讨厌的人
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
    if (len(character_data.social_contact[0]) + len(character_data.social_contact[1]) + len(character_data.social_contact[2]) + len(character_data.social_contact[3])) > 0:
        return 1
    return 0
