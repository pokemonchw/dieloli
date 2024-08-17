from Script.Design import settle_behavior, constant
from Script.Core import cache_control, game_type


cache: game_type.Cache = cache_control.cache


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_HIT_POINT)
def handle_target_add_small_hit_point(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    交互对象增加少量健康
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    add_hit_point = add_time * 10
    target_data.hit_point += add_hit_point
    if target_data.hit_point > target_data.hit_point_max:
        add_hit_point -= target_data.hit_point - target_data.hit_point_max
        target_data.hit_point = target_data.hit_point_max


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_HIT_POINT)
def handle_target_add_medium_hit_point(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    交互对象增加中量健康
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    add_hit_point = add_time * 50
    target_data.hit_point += add_hit_point
    if target_data.hit_point > target_data.hit_point_max:
        add_hit_point -= target_data.hit_point - target_data.hit_point_max
        target_data.hit_point = target_data.hit_point_max


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_HIT_POINT)
def handle_target_add_large_hit_point(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    交互对象增加大量健康
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    add_hit_point = add_time * 100
    target_data.hit_point += add_hit_point
    if target_data.hit_point > target_data.hit_point_max:
        add_hit_point -= target_data.hit_point - target_data.hit_point_max
        target_data.hit_point = target_data.hit_point_max


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_SMALL_HIT_POINT)
def handle_target_sub_small_hit_point(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    交互对象减少少量健康
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    sub_hit_point = add_time * 10
    if target_data.hit_point >= sub_hit_point:
        target_data.hit_point -= sub_hit_point
    else:
        character_data.hit_point = 0


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_MEDIUM_HIT_POINT)
def handle_target_sub_medium_hit_point(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    交互对象减少中量健康
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    sub_hit_point = add_time * 50
    if target_data.hit_point >= sub_hit_point:
        target_data.hit_point -= sub_hit_point
    else:
        character_data.hit_point = 0


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_LARGE_HIT_POINT)
def handle_target_sub_large_hit_point(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    交互对象减少大量健康
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    sub_hit_point = add_time * 100
    if target_data.hit_point >= sub_hit_point:
        target_data.hit_point -= sub_hit_point
    else:
        character_data.hit_point = 0


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_MANA_POINT)
def handle_target_add_small_mana_point(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    交互对象增加少量体力
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    add_mana_point = add_time * 10
    target_data.mana_point += add_mana_point
    if target_data.mana_point > target_data.mana_point_max:
        add_mana_point -= target_data.mana_point - target_data.mana_point_max
        target_data.mana_point = target_data.mana_point_max


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_MANA_POINT)
def handle_target_add_medium_mana_point(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    交互对象增加中量体力
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    add_mana_point = add_time * 50
    target_data.mana_point += add_mana_point
    if target_data.mana_point > target_data.mana_point_max:
        add_mana_point -= target_data.mana_point - target_data.mana_point_max
        target_data.mana_point = target_data.mana_point_max


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_MANA_POINT)
def handle_target_add_large_mana_point(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    交互对象增加大量体力
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    add_mana_point = add_time * 100
    target_data.mana_point += add_mana_point
    if target_data.mana_point > target_data.mana_point_max:
        add_mana_point -= target_data.mana_point - target_data.mana_point_max
        target_data.mana_point = target_data.mana_point_max


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_SMALL_MANA_POINT)
def handle_target_sub_small_mana_point(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    交互对象减少少量体力
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    sub_mana_point = add_time * 10
    if target_data.mana_point >= sub_mana_point:
        target_data.mana_point -= sub_mana_point
    else:
        sub_mana_point -= target_data.mana_point
        target_data.mana_point = 0
        sub_hit_point = sub_mana_point / 15
        if sub_hit_point > target_data.hit_point:
            sub_hit_point = target_data.hit_point
        target_data.hit_point -= sub_hit_point


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_MEDIUM_MANA_POINT)
def handle_target_sub_medium_mana_point(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    交互对象减少中量体力
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    sub_mana_point = add_time * 50
    if target_data.mana_point >= sub_mana_point:
        target_data.mana_point -= sub_mana_point
    else:
        sub_mana_point -= target_data.mana_point
        target_data.mana_point = 0
        sub_hit_point = sub_mana_point / 15
        if sub_hit_point > target_data.hit_point:
            sub_hit_point = target_data.hit_point
        target_data.hit_point -= sub_hit_point


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_LARGE_MANA_POINT)
def handle_target_sub_large_mana_point(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    交互对象减少大量体力
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    sub_mana_point = add_time * 100
    if target_data.mana_point >= sub_mana_point:
        target_data.mana_point -= sub_mana_point
    else:
        sub_mana_point -= target_data.mana_point
        target_data.mana_point = 0
        sub_hit_point = sub_mana_point / 15
        if sub_hit_point > target_data.hit_point:
            sub_hit_point = target_data.hit_point
        target_data.hit_point -= sub_hit_point


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_ACHE)
def handle_target_add_small_ache(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量疼痛
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    target_data.status.setdefault(23,0)
    now_value = target_data.status[23]
    now_add_value = (1 + now_value / 100) * add_time
    target_data.status[23] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(23,0)
    target_change.status[23] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_ACHE)
def handle_target_add_medium_ache(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量疼痛
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    target_data.status.setdefault(23,0)
    now_value = target_data.status[23]
    now_add_value = (1 + now_value / 50) * add_time
    target_data.status[23] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(23,0)
    target_change.status[23] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_ACHE)
def handle_target_add_large_ache(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量疼痛
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    target_data.status.setdefault(23,0)
    now_value = target_data.status[23]
    now_add_value = (1 + now_value / 10) * add_time
    target_data.status[23] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(23,0)
    target_change.status[23] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_SMALL_ACHE)
def handle_target_sub_small_ache(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少少量疼痛
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    target_data.status.setdefault(23,0)
    if not target_data.status[23]:
        return
    now_value = target_data.status[23]
    now_sub_value = (1 + now_value / 100) * add_time
    target_data.status[23] -= now_sub_value
    target_data.status[23] = max(target_data.status[23],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(23,0)
    target_change.status[23] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_MEDIUM_ACHE)
def handle_target_sub_medium_ache(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少中量疼痛
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    target_data.status.setdefault(23,0)
    if not target_data.status[23]:
        return
    now_value = target_data.status[23]
    now_sub_value = (1 + now_value / 50) * add_time
    target_data.status[23] -= now_sub_value
    target_data.status[23] = max(target_data.status[23],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(23,0)
    target_change.status[23] += now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_LARGE_ACHE)
def handle_target_sub_large_ache(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少大量疼痛
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    target_data.status.setdefault(23,0)
    if not target_data.status[23]:
        return
    now_value = target_data.status[23]
    now_sub_value = (1 + now_value / 10) * add_time
    target_data.status[23] -= now_sub_value
    target_data.status[23] = max(target_data.status[23],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(23,0)
    target_change.status[23] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_VERTIGO)
def handle_target_add_small_vertigo(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量眩晕
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    target_data.status.setdefault(24,0)
    now_value = target_data.status[24]
    now_add_value = (1 + now_value / 100) * add_time
    target_data.status[24] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(24,0)
    target_change.status[24] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_VERTIGO)
def handle_target_add_medium_vertigo(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量眩晕
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    target_data.status.setdefault(24,0)
    now_value = target_data.status[24]
    now_add_value = (1 + now_value / 50) * add_time
    target_data.status[24] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(24,0)
    target_change.status[24] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_VERTIGO)
def handle_target_add_large_vertigo(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量眩晕
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    target_data.status.setdefault(24,0)
    now_value = target_data.status[24]
    now_add_value = (1 + now_value / 10) * add_time
    target_data.status[24] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(24,0)
    target_change.status[24] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_SMALL_VERTIGO)
def handle_target_sub_small_vertigo(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少少量眩晕
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    target_data.status.setdefault(24,0)
    if not target_data.status[24]:
        return
    now_value = target_data.status[24]
    now_sub_value = (1 + now_value / 100) * add_time
    target_data.status[24] -= now_sub_value
    target_data.status[24] = max(target_data.status[24],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(24,0)
    target_change.status[24] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_MEDIUM_VERTIGO)
def handle_target_sub_medium_vertigo(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少中量眩晕
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    target_data.status.setdefault(24,0)
    if not target_data.status[24]:
        return
    now_value = target_data.status[24]
    now_sub_value = (1 + now_value / 50) * add_time
    target_data.status[24] -= now_sub_value
    target_data.status[24] = max(target_data.status[24],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(24,0)
    target_change.status[24] += now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_LARGE_VERTIGO)
def handle_target_sub_large_vertigo(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少大量眩晕
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    target_data.status.setdefault(24,0)
    if not target_data.status[24]:
        return
    now_value = target_data.status[24]
    now_sub_value = (1 + now_value / 10) * add_time
    target_data.status[24] -= now_sub_value
    target_data.status[24] = max(target_data.status[24],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(24,0)
    target_change.status[24] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_SMALL_TIRED)
def handle_target_sub_small_tired(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少少量疲惫
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    target_data.status.setdefault(25,0)
    if not target_data.status[25]:
        return
    now_value = target_data.status[25]
    now_sub_value = (1 + now_value / 100) * add_time / 10
    target_data.status[25] -= now_sub_value
    target_data.status[25] = max(target_data.status[25],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(25,0)
    target_change.status[25] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_MEDIUM_TIRED)
def handle_target_sub_medium_tired(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少中量疲惫
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    target_data.status.setdefault(25,0)
    if not target_data.status[25]:
        return
    now_value = target_data.status[25]
    now_sub_value = (1 + now_value / 50) * add_time / 10
    target_data.status[25] -= now_sub_value
    target_data.status[25] = max(target_data.status[25],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(25,0)
    target_change.status[25] += now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_LARGE_TIRED)
def handle_target_sub_large_tired(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少大量疲惫
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    target_data.status.setdefault(25,0)
    if not target_data.status[25]:
        return
    now_value = target_data.status[25]
    now_sub_value = (1 + now_value / 10) * add_time / 10
    target_data.status[25] -= now_sub_value
    target_data.status[25] = max(target_data.status[25],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(25,0)
    target_change.status[25] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_INTOXICATED)
def handle_target_add_small_intoxicated(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量迷醉
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    target_data.status.setdefault(26,0)
    now_value = target_data.status[26]
    now_add_value = (1 + now_value / 100) * add_time
    target_data.status[26] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(26,0)
    target_change.status[26] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_INTOXICATED)
def handle_target_add_medium_intoxicated(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量迷醉
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    target_data.status.setdefault(26,0)
    now_value = target_data.status[26]
    now_add_value = (1 + now_value / 50) * add_time
    target_data.status[26] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(26,0)
    target_change.status[26] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_INTOXICATED)
def handle_target_add_large_intoxicated(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量迷醉
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    target_data.status.setdefault(26,0)
    now_value = target_data.status[26]
    now_add_value = (1 + now_value / 10) * add_time
    target_data.status[26] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(26,0)
    target_change.status[26] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_SMALL_INTOXICATED)
def handle_target_sub_small_intoxicated(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少少量迷醉
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    target_data.status.setdefault(26,0)
    if not target_data.status[26]:
        return
    now_value = target_data.status[26]
    now_sub_value = (1 + now_value / 100) * add_time
    target_data.status[26] -= now_sub_value
    target_data.status[26] = max(target_data.status[26],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(26,0)
    target_change.status[26] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_MEDIUM_INTOXICATED)
def handle_target_sub_medium_intoxicated(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少中量迷醉
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    target_data.status.setdefault(26,0)
    if not target_data.status[26]:
        return
    now_value = target_data.status[26]
    now_sub_value = (1 + now_value / 50) * add_time
    target_data.status[26] -= now_sub_value
    target_data.status[26] = max(target_data.status[26],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(26,0)
    target_change.status[26] += now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_LARGE_INTOXICATED)
def handle_target_sub_large_intoxicated(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少大量迷醉
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
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    target_data.status.setdefault(26,0)
    if not target_data.status[26]:
        return
    now_value = target_data.status[26]
    now_sub_value = (1 + now_value / 10) * add_time
    target_data.status[26] -= now_sub_value
    target_data.status[26] = max(target_data.status[26],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(26,0)
    target_change.status[26] -= now_sub_value
