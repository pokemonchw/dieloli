from Script.Design import settle_behavior
from Script.Core import cache_control, constant, game_type


cache: game_type.Cache = cache_control.cache


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_HIT_POINT)
def handle_add_small_hit_point(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    增加少量体力
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    add_hit_point = add_time * 10
    character_data.hit_point += add_hit_point
    if character_data.hit_point > character_data.hit_point_max:
        add_hit_point -= character_data.hit_point - character_data.hit_point_max
        character_data.hit_point = character_data.hit_point_max
    change_data.hit_point += add_hit_point


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_HIT_POINT)
def handle_add_medium_hit_point(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    增加中量体力
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    add_hit_point = add_time * 50
    character_data.hit_point += add_hit_point
    if character_data.hit_point > character_data.hit_point_max:
        add_hit_point -= character_data.hit_point - character_data.hit_point_max
        character_data.hit_point = character_data.hit_point_max
    change_data.hit_point += add_hit_point


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_HIT_POINT)
def handle_add_large_hit_point(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    增加大量体力
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    add_hit_point = add_time * 100
    character_data.hit_point += add_hit_point
    if character_data.hit_point > character_data.hit_point_max:
        add_hit_point -= character_data.hit_point - character_data.hit_point_max
        character_data.hit_point = character_data.hit_point_max
    change_data.hit_point += add_hit_point


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_SMALL_HIT_POINT)
def handle_sub_small_hit_point(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    减少少量体力
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    sub_hit_point = add_time * 10
    if character_data.hit_point >= sub_hit_point:
        character_data.hit_point -= sub_hit_point
        change_data.hit_point -= sub_hit_point
    else:
        change_data.hit_point -= character_data.hit_point
        character_data.hit_point = 0


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_MEDIUM_HIT_POINT)
def handle_sub_medium_hit_point(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    减少中量体力
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    sub_hit_point = add_time * 50
    if character_data.hit_point >= sub_hit_point:
        character_data.hit_point -= sub_hit_point
        change_data.hit_point -= sub_hit_point
    else:
        change_data.hit_point -= character_data.hit_point
        character_data.hit_point = 0


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_LARGE_HIT_POINT)
def handle_sub_large_hit_point(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    减少大量体力
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    sub_hit_point = add_time * 100
    if character_data.hit_point >= sub_hit_point:
        character_data.hit_point -= sub_hit_point
        change_data.hit_point -= sub_hit_point
    else:
        change_data.hit_point -= character_data.hit_point
        character_data.hit_point = 0


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_MANA_POINT)
def handle_add_small_mana_point(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    增加少量气力
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    add_mana_point = add_time * 10
    character_data.mana_point += add_mana_point
    if character_data.mana_point > character_data.mana_point_max:
        add_mana_point -= character_data.mana_point - character_data.mana_point_max
        character_data.mana_point = character_data.mana_point_max
    change_data.mana_point += add_mana_point


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_MANA_POINT)
def handle_add_medium_mana_point(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    增加中量气力
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    add_mana_point = add_time * 50
    character_data.mana_point += add_mana_point
    if character_data.mana_point > character_data.mana_point_max:
        add_mana_point -= character_data.mana_point - character_data.mana_point_max
        character_data.mana_point = character_data.mana_point_max
    change_data.mana_point += add_mana_point



@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_MANA_POINT)
def handle_add_large_mana_point(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    增加大量气力
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    add_mana_point = add_time * 100
    character_data.mana_point += add_mana_point
    if character_data.mana_point > character_data.mana_point_max:
        add_mana_point -= character_data.mana_point - character_data.mana_point_max
        character_data.mana_point = character_data.mana_point_max
    change_data.mana_point += add_mana_point



@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_SMALL_MANA_POINT)
def handle_sub_small_mana_point(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    减少少量气力
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    sub_mana_point = add_time * 10
    if character_data.mana_point >= sub_mana_point:
        character_data.mana_point -= sub_mana_point
        change_data.mana_point -= sub_mana_point
    else:
        change_data.mana_point -= character_data.mana_point
        sub_mana_point -= character_data.mana_point
        character_data.mana_point = 0
        sub_hit_point = sub_mana_point / 15
        if sub_hit_point > character_data.hit_point:
            sub_hit_point = character_data.hit_point
        character_data.hit_point -= sub_hit_point
        change_data.hit_point -= sub_hit_point


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_MEDIUM_MANA_POINT)
def handle_sub_medium_mana_point(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    减少中量气力
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    sub_mana_point = add_time * 50
    if character_data.mana_point >= sub_mana_point:
        character_data.mana_point -= sub_mana_point
        change_data.mana_point -= sub_mana_point
    else:
        change_data.mana_point -= character_data.mana_point
        sub_mana_point -= character_data.mana_point
        character_data.mana_point = 0
        sub_hit_point = sub_mana_point / 15
        if sub_hit_point > character_data.hit_point:
            sub_hit_point = character_data.hit_point
        character_data.hit_point -= sub_hit_point
        change_data.hit_point -= sub_hit_point


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_LARGE_MANA_POINT)
def handle_sub_large_mana_point(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    减少大量气力
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    sub_mana_point = add_time * 100
    if character_data.mana_point >= sub_mana_point:
        character_data.mana_point -= sub_mana_point
        change_data.mana_point -= sub_mana_point
    else:
        change_data.mana_point -= character_data.mana_point
        sub_mana_point -= character_data.mana_point
        character_data.mana_point = 0
        sub_hit_point = sub_mana_point / 15
        if sub_hit_point > character_data.hit_point:
            sub_hit_point = character_data.hit_point
        character_data.hit_point -= sub_hit_point
        change_data.hit_point -= sub_hit_point


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_ACHE)
def handle_add_small_ache(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量疼痛
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(23,0)
    now_value = character_data.status[23]
    now_add_value = (1 + now_value / 100) * add_time
    character_data.status[23] += now_add_value
    change_data.status.setdefault(23,0)
    change_data.status[23] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_ACHE)
def handle_add_medium_ache(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量疼痛
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(23,0)
    now_value = character_data.status[23]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[23] += now_add_value
    change_data.status.setdefault(23,0)
    change_data.status[23] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_ACHE)
def handle_add_large_ache(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量疼痛
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(23,0)
    now_value = character_data.status[23]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[23] += now_add_value
    change_data.status.setdefault(23,0)
    change_data.status[23] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_SMALL_ACHE)
def handle_sub_small_ache(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少少量疼痛
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(23,0)
    now_value = character_data.status[23]
    now_sub_value = (1 + now_value / 100) * add_time
    character_data.status[23] -= now_sub_value
    character_data.status[23] = max(character_data.status[23],0)
    change_data.status.setdefault(23,0)
    change_data.status[23] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_MEDIUM_ACHE)
def handle_sub_medium_ache(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少中量疼痛
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(23,0)
    now_value = character_data.status[23]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[23] -= now_sub_value
    character_data.status[23] = max(character_data.status[23],0)
    change_data.status.setdefault(23,0)
    change_data.status[23] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_LARGE_ACHE)
def handle_sub_large_ache(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少大量疼痛
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(23,0)
    now_value = character_data.status[23]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[23] -= now_sub_value
    character_data.status[23] = max(character_data.status[23],0)
    change_data.status.setdefault(23,0)
    change_data.status[23] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_VERTIGO)
def handle_add_small_vertigo(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量眩晕
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(24,0)
    now_value = character_data.status[24]
    now_add_value = (1 + now_value / 100) * add_time
    character_data.status[24] += now_add_value
    change_data.status.setdefault(24,0)
    change_data.status[24] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_VERTIGO)
def handle_add_medium_vertigo(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量眩晕
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(24,0)
    now_value = character_data.status[24]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[24] += now_add_value
    change_data.status.setdefault(24,0)
    change_data.status[24] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_VERTIGO)
def handle_add_large_vertigo(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量眩晕
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(24,0)
    now_value = character_data.status[24]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[24] += now_add_value
    change_data.status.setdefault(24,0)
    change_data.status[24] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_SMALL_VERTIGO)
def handle_sub_small_vertigo(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少少量眩晕
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(24,0)
    now_value = character_data.status[24]
    now_sub_value = (1 + now_value / 100) * add_time
    character_data.status[24] -= now_sub_value
    character_data.status[24] = max(character_data.status[24],0)
    change_data.status.setdefault(24,0)
    change_data.status[24] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_MEDIUM_VERTIGO)
def handle_sub_medium_vertigo(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少中量眩晕
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(24,0)
    now_value = character_data.status[24]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[24] -= now_sub_value
    character_data.status[24] = max(character_data.status[24],0)
    change_data.status.setdefault(24,0)
    change_data.status[24] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_LARGE_VERTIGO)
def handle_sub_large_vertigo(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少大量眩晕
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(24,0)
    now_value = character_data.status[24]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[24] -= now_sub_value
    character_data.status[24] = max(character_data.status[24],0)
    change_data.status.setdefault(24,0)
    change_data.status[24] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_TIRED)
def handle_add_small_tired(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量疲惫
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(25,0)
    now_value = character_data.status[25]
    now_add_value = (1 + now_value / 100) * add_time
    character_data.status[25] += now_add_value
    change_data.status.setdefault(25,0)
    change_data.status[25] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_TIRED)
def handle_add_medium_tired(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量疲惫
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(25,0)
    now_value = character_data.status[25]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[25] += now_add_value
    change_data.status.setdefault(25,0)
    change_data.status[25] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_TIRED)
def handle_add_large_tired(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量疲惫
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(25,0)
    now_value = character_data.status[25]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[25] += now_add_value
    change_data.status.setdefault(25,0)
    change_data.status[25] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_SMALL_TIRED)
def handle_sub_small_tired(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少少量疲惫
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(25,0)
    now_value = character_data.status[25]
    now_sub_value = (1 + now_value / 100) * add_time
    character_data.status[25] -= now_sub_value
    character_data.status[25] = max(character_data.status[25],0)
    change_data.status.setdefault(25,0)
    change_data.status[25] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_MEDIUM_TIRED)
def handle_sub_medium_tired(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少中量疲惫
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(25,0)
    now_value = character_data.status[25]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[25] -= now_sub_value
    character_data.status[25] = max(character_data.status[25],0)
    change_data.status.setdefault(25,0)
    change_data.status[25] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_LARGE_TIRED)
def handle_sub_large_tired(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少大量疲惫
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(25,0)
    now_value = character_data.status[25]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[25] -= now_sub_value
    character_data.status[25] = max(character_data.status[25],0)
    change_data.status.setdefault(25,0)
    change_data.status[25] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_INTOXICATED)
def handle_add_small_intoxicated(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量迷醉
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(26,0)
    now_value = character_data.status[26]
    now_add_value = (1 + now_value / 100) * add_time
    character_data.status[26] += now_add_value
    change_data.status.setdefault(26,0)
    change_data.status[26] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_INTOXICATED)
def handle_add_medium_intoxicated(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量迷醉
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(26,0)
    now_value = character_data.status[26]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[26] += now_add_value
    change_data.status.setdefault(26,0)
    change_data.status[26] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_INTOXICATED)
def handle_add_large_intoxicated(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量迷醉
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(26,0)
    now_value = character_data.status[26]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[26] += now_add_value
    change_data.status.setdefault(26,0)
    change_data.status[26] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_SMALL_INTOXICATED)
def handle_sub_small_intoxicated(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少少量迷醉
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(26,0)
    now_value = character_data.status[26]
    now_sub_value = (1 + now_value / 100) * add_time
    character_data.status[26] -= now_sub_value
    character_data.status[26] = max(character_data.status[26],0)
    change_data.status.setdefault(26,0)
    change_data.status[26] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_MEDIUM_INTOXICATED)
def handle_sub_medium_intoxicated(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少中量迷醉
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(26,0)
    now_value = character_data.status[26]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[26] -= now_sub_value
    character_data.status[26] = max(character_data.status[26],0)
    change_data.status.setdefault(26,0)
    change_data.status[26] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_LARGE_INTOXICATED)
def handle_sub_large_intoxicated(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少大量迷醉
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(26,0)
    now_value = character_data.status[26]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[26] -= now_sub_value
    character_data.status[26] = max(character_data.status[26],0)
    change_data.status.setdefault(26,0)
    change_data.status[26] -= now_sub_value
