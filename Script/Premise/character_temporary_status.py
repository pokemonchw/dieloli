from Script.Design import handle_premise, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.HAVE_TARGET)
def handle_have_target(character_id: int) -> int:
    """
    校验角色是否有交互对象
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    if character_data.target_character_id == character_id:
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.EAT_SPRING_FOOD)
def handle_eat_spring_food(character_id: int) -> int:
    """
    校验角色是否正在食用春药品质的食物
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    if character_data.behavior.food_quality == 4:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.EAT_GOOD_FOOD)
def handle_eat_good_food(character_id: int) -> int:
    """
    校验角色是否正在食用良好品质的食物
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    if character_data.behavior.food_quality > 1:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.EAT_RUBBISH_FOOD)
def handle_eat_rubbish_food(character_id: int) -> int:
    """
    校验角色是否正在食用垃圾品质的食物
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.behavior.food_quality < 2:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_IS_LIVE)
def handle_target_is_live(character_id: int) -> int:
    """
    校验交互对象是否未死亡
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if not target_data.dead:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.NO_FOLLOW)
def handle_no_follow(character_id: int) -> int:
    """
    判断角色是否未处于跟随状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.follow == -1:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_LOSE_FIRST_KISS)
def handle_is_lose_first_kiss(character_id: int) -> int:
    """
    校验角色是否正在失去初吻
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.behavior.temporary_status.lose_first_kiss:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_IS_LOSE_FIRST_KISS)
def handle_target_is_lose_first_kiss(character_id: int) -> int:
    """
    校验交互对象是否正在失去初吻
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.behavior.temporary_status.lose_first_kiss:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_IS_MOVING)
def handle_target_is_moving(character_id: int) -> int:
    """
    校验交互对象是否正在移动
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.state == constant.CharacterStatus.STATUS_MOVE:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_IS_PASSIVE_SEX)
def handle_target_is_passive_sex(character_id: int) -> int:
    """
    校验交互对象是否处于做爱受体状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.passive_sex


@handle_premise.add_premise(constant.Premise.TARGET_NOT_PASSIVE_SEX)
def handle_target_not_passive_sex(character_id: int) -> int:
    """
    校验交互对象是否未处于做爱受体状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 1
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.passive_sex == 1:
        return 0
    return 1
