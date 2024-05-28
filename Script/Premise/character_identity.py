from Script.Design import handle_premise, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.TARGET_NO_PLAYER)
def handle_target_no_player(character_id: int) -> int:
    """
    校验角色目标对像是否不是玩家
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    if character_data.target_character_id > 0:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_STUDENT)
def handle_is_student(character_id: int) -> int:
    """
    校验角色是否是学生
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    if character_data.age <= 18:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_TEACHER)
def handle_is_teacher(character_id: int) -> int:
    """
    校验角色是否是老师
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    return character_id in cache.teacher_school_timetable


@handle_premise.add_premise(constant.Premise.TARGET_IS_PLAYER)
def handle_target_is_player(character_id: int) -> int:
    """
    校验角色目标是否是玩家
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    if character_data.target_character_id == 0:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_PLAYER)
def handle_is_player(character_id: int) -> int:
    """
    校验是否是玩家角色
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    if not character_id:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.NO_PLAYER)
def handle_no_player(character_id: int) -> int:
    """
    校验是否不是玩家角色
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    if character_id:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_IS_STUDENT)
def handle_target_is_student(character_id: int) -> int:
    """
    校验交互对象是否是学生
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.age <= 18


@handle_premise.add_premise(constant.Premise.IS_PRIMARY_SCHOOL_STUDENTS)
def handle_is_primary_school_students(character_id: int) -> int:
    """
    校验角色是否是小学生
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    return character_data.age <= 12


@handle_premise.add_premise(constant.Premise.IS_PLAYER_TARGET)
def handle_is_player_target(character_id: int) -> int:
    """
    校验角色是否是玩家的交互对象
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    if not character_id:
        return 0
    player_data: game_type.Character = cache.character_data[0]
    return player_data.target_character_id == character_id


@handle_premise.add_premise(constant.Premise.IS_JOINED_CLUB)
def handle_is_joined_club(character_id: int) -> int:
    """
    校验角色是否已经加入了社团
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    return 2 in character_data.identity_data


@handle_premise.add_premise(constant.Premise.NOT_JOINED_CLUB)
def handle_not_joined_club(character_id: int) -> int:
    """
    校验角色是否没有加入社团
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    return 2 not in character_data.identity_data

