from Script.Design import settle_behavior
from Script.Core import cache_control, constant, game_type


cache: game_type.Cache = cache_control.cache


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_HAPPY)
def handle_target_add_small_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量快乐
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
    target_data.status.setdefault(8,0)
    now_value = target_data.status[8]
    now_add_value = (1 + now_value / 100) * add_time
    target_data.status[8] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(8,0)
    target_change.status[8] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_HAPPY)
def handle_target_add_medium_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量快乐
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
    target_data.status.setdefault(8,0)
    now_value = target_data.status[8]
    now_add_value = (1 + now_value / 50) * add_time
    target_data.status[8] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(8,0)
    target_change.status[8] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_HAPPY)
def handle_target_add_large_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量快乐
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
    target_data.status.setdefault(8,0)
    now_value = target_data.status[8]
    now_add_value = (1 + now_value / 10) * add_time
    target_data.status[8] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(8,0)
    target_change.status[8] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_SMALL_HAPPY)
def handle_target_sub_small_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少少量快乐
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
    target_data.status.setdefault(8,0)
    now_value = target_data.status[8]
    now_sub_value = (1 + now_value / 100) * add_time
    target_data.status[8] -= now_sub_value
    target_data.status[8] = max(target_data.status[8],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(8,0)
    target_change.status[8] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_MEDIUM_HAPPY)
def handle_target_sub_medium_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少中量快乐
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
    target_data.status.setdefault(8,0)
    now_value = target_data.status[8]
    now_sub_value = (1 + now_value / 50) * add_time
    target_data.status[8] -= now_sub_value
    target_data.status[8] = max(target_data.status[8],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(8,0)
    target_change.status[8] += now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_LARGE_HAPPY)
def handle_target_sub_large_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少大量快乐
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
    target_data.status.setdefault(8,0)
    now_value = target_data.status[8]
    now_sub_value = (1 + now_value / 10) * add_time
    target_data.status[8] -= now_sub_value
    target_data.status[8] = max(target_data.status[8],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(8,0)
    target_change.status[8] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_PAIN)
def handle_target_add_small_pain(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量痛苦
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
    target_data.status.setdefault(9,0)
    now_value = target_data.status[9]
    now_add_value = (1 + now_value / 100) * add_time
    target_data.status[9] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(9,0)
    target_change.status[9] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_PAIN)
def handle_target_add_medium_pain(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量痛苦
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
    target_data.status.setdefault(9,0)
    now_value = target_data.status[9]
    now_add_value = (1 + now_value / 50) * add_time
    target_data.status[9] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(9,0)
    target_change.status[9] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_PAIN)
def handle_target_add_large_pain(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量痛苦
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
    target_data.status.setdefault(9,0)
    now_value = target_data.status[9]
    now_add_value = (1 + now_value / 10) * add_time
    target_data.status[9] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(9,0)
    target_change.status[9] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_SMALL_PAIN)
def handle_target_sub_small_pain(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少少量痛苦
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
    target_data.status.setdefault(9,0)
    now_value = target_data.status[9]
    now_sub_value = (1 + now_value / 100) * add_time
    target_data.status[9] -= now_sub_value
    target_data.status[9] = max(target_data.status[9],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(9,0)
    target_change.status[9] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_MEDIUM_PAIN)
def handle_target_sub_medium_pain(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少中量痛苦
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
    target_data.status.setdefault(9,0)
    now_value = target_data.status[9]
    now_sub_value = (1 + now_value / 50) * add_time
    target_data.status[9] -= now_sub_value
    target_data.status[9] = max(target_data.status[9],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(9,0)
    target_change.status[9] += now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_LARGE_PAIN)
def handle_target_sub_large_pain(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少大量痛苦
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
    target_data.status.setdefault(9,0)
    now_value = target_data.status[9]
    now_sub_value = (1 + now_value / 10) * add_time
    target_data.status[9] -= now_sub_value
    target_data.status[9] = max(target_data.status[9],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(9,0)
    target_change.status[9] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_YEARN)
def handle_target_add_small_yearn(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量渴望
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
    target_data.status.setdefault(10,0)
    now_value = target_data.status[10]
    now_add_value = (1 + now_value / 100) * add_time
    target_data.status[10] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(10,0)
    target_change.status[10] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_YEARN)
def handle_target_add_medium_yearn(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量渴望
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
    target_data.status.setdefault(10,0)
    now_value = target_data.status[10]
    now_add_value = (1 + now_value / 50) * add_time
    target_data.status[10] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(10,0)
    target_change.status[10] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_YEARN)
def handle_target_add_large_yearn(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量渴望
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
    target_data.status.setdefault(10,0)
    now_value = target_data.status[10]
    now_add_value = (1 + now_value / 10) * add_time
    target_data.status[10] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(10,0)
    target_change.status[10] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_SMALL_YEARN)
def handle_target_sub_small_yearn(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少少量渴望
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
    target_data.status.setdefault(10,0)
    now_value = target_data.status[10]
    now_sub_value = (1 + now_value / 100) * add_time
    target_data.status[10] -= now_sub_value
    target_data.status[10] = max(target_data.status[10],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(10,0)
    target_change.status[10] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_MEDIUM_YEARN)
def handle_target_sub_medium_yearn(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少中量渴望
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
    target_data.status.setdefault(10,0)
    now_value = target_data.status[10]
    now_sub_value = (1 + now_value / 50) * add_time
    target_data.status[10] -= now_sub_value
    target_data.status[10] = max(target_data.status[10],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(10,0)
    target_change.status[10] += now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_LARGE_YEARN)
def handle_target_sub_large_yearn(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少大量渴望
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
    target_data.status.setdefault(10,0)
    now_value = target_data.status[10]
    now_sub_value = (1 + now_value / 10) * add_time
    target_data.status[10] -= now_sub_value
    target_data.status[10] = max(target_data.status[10],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(10,0)
    target_change.status[10] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_FEAR)
def handle_target_add_small_fear(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量恐惧
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
    target_data.status.setdefault(11,0)
    now_value = target_data.status[11]
    now_add_value = (1 + now_value / 100) * add_time
    target_data.status[11] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(11,0)
    target_change.status[11] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_FEAR)
def handle_target_add_medium_fear(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量恐惧
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
    target_data.status.setdefault(11,0)
    now_value = target_data.status[11]
    now_add_value = (1 + now_value / 50) * add_time
    target_data.status[11] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(11,0)
    target_change.status[11] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_FEAR)
def handle_target_add_large_fear(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量恐惧
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
    target_data.status.setdefault(11,0)
    now_value = target_data.status[11]
    now_add_value = (1 + now_value / 10) * add_time
    target_data.status[11] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(11,0)
    target_change.status[11] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_SMALL_FEAR)
def handle_target_sub_small_fear(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少少量恐惧
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
    target_data.status.setdefault(11,0)
    now_value = target_data.status[11]
    now_sub_value = (1 + now_value / 100) * add_time
    target_data.status[11] -= now_sub_value
    target_data.status[11] = max(target_data.status[11],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(11,0)
    target_change.status[11] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_MEDIUM_FEAR)
def handle_target_sub_medium_fear(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少中量恐惧
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
    target_data.status.setdefault(11,0)
    now_value = target_data.status[11]
    now_sub_value = (1 + now_value / 50) * add_time
    target_data.status[11] -= now_sub_value
    target_data.status[11] = max(target_data.status[11],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(11,0)
    target_change.status[11] += now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_LARGE_FEAR)
def handle_target_sub_large_fear(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少大量恐惧
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
    target_data.status.setdefault(11,0)
    now_value = target_data.status[11]
    now_sub_value = (1 + now_value / 10) * add_time
    target_data.status[11] -= now_sub_value
    target_data.status[11] = max(target_data.status[11],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(11,0)
    target_change.status[11] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_ANTIPATHY)
def handle_target_add_small_antipathy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量反感
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
    target_data.status.setdefault(12,0)
    now_value = target_data.status[12]
    now_add_value = (1 + now_value / 100) * add_time
    target_data.status[12] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(12,0)
    target_change.status[12] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_ANTIPATHY)
def handle_target_add_medium_antipathy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量反感
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
    target_data.status.setdefault(12,0)
    now_value = target_data.status[12]
    now_add_value = (1 + now_value / 50) * add_time
    target_data.status[12] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(12,0)
    target_change.status[12] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_ANTIPATHY)
def handle_target_add_large_antipathy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量反感
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
    target_data.status.setdefault(12,0)
    now_value = target_data.status[12]
    now_add_value = (1 + now_value / 10) * add_time
    target_data.status[12] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(12,0)
    target_change.status[12] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_SMALL_ANTIPATHY)
def handle_target_sub_small_antipathy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少少量反感
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
    target_data.status.setdefault(12,0)
    now_value = target_data.status[12]
    now_sub_value = (1 + now_value / 100) * add_time
    target_data.status[12] -= now_sub_value
    target_data.status[12] = max(target_data.status[12],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(12,0)
    target_change.status[12] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_MEDIUM_ANTIPATHY)
def handle_target_sub_medium_antipathy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少中量反感
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
    target_data.status.setdefault(12,0)
    now_value = target_data.status[12]
    now_sub_value = (1 + now_value / 50) * add_time
    target_data.status[12] -= now_sub_value
    target_data.status[12] = max(target_data.status[12],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(12,0)
    target_change.status[12] += now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_LARGE_ANTIPATHY)
def handle_target_sub_large_antipathy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少大量反感
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
    target_data.status.setdefault(12,0)
    now_value = target_data.status[12]
    now_sub_value = (1 + now_value / 10) * add_time
    target_data.status[12] -= now_sub_value
    target_data.status[12] = max(target_data.status[12],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(12,0)
    target_change.status[12] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_SHAME)
def handle_target_add_small_shame(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量羞耻
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
    target_data.status.setdefault(13,0)
    now_value = target_data.status[13]
    now_add_value = (1 + now_value / 100) * add_time
    target_data.status[13] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(13,0)
    target_change.status[13] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_SHAME)
def handle_target_add_medium_shame(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量羞耻
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
    target_data.status.setdefault(13,0)
    now_value = target_data.status[13]
    now_add_value = (1 + now_value / 50) * add_time
    target_data.status[13] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(13,0)
    target_change.status[13] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_SHAME)
def handle_target_add_large_shame(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量羞耻
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
    target_data.status.setdefault(13,0)
    now_value = target_data.status[13]
    now_add_value = (1 + now_value / 10) * add_time
    target_data.status[13] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(13,0)
    target_change.status[13] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_SMALL_SHAME)
def handle_target_sub_small_shame(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少少量羞耻
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
    target_data.status.setdefault(13,0)
    now_value = target_data.status[13]
    now_sub_value = (1 + now_value / 100) * add_time
    target_data.status[13] -= now_sub_value
    target_data.status[13] = max(target_data.status[13],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(13,0)
    target_change.status[13] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_MEDIUM_SHAME)
def handle_target_sub_medium_shame(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少中量羞耻
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
    target_data.status.setdefault(13,0)
    now_value = target_data.status[13]
    now_sub_value = (1 + now_value / 50) * add_time
    target_data.status[13] -= now_sub_value
    target_data.status[13] = max(target_data.status[13],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(13,0)
    target_change.status[13] += now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_LARGE_SHAME)
def handle_target_sub_large_shame(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少大量羞耻
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
    target_data.status.setdefault(13,0)
    now_value = target_data.status[13]
    now_sub_value = (1 + now_value / 10) * add_time
    target_data.status[13] -= now_sub_value
    target_data.status[13] = max(target_data.status[13],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(13,0)
    target_change.status[13] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_DEPRESSED)
def handle_target_add_small_depressed(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量抑郁
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
    target_data.status.setdefault(14,0)
    now_value = target_data.status[14]
    now_add_value = (1 + now_value / 100) * add_time
    target_data.status[14] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(14,0)
    target_change.status[14] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_DEPRESSED)
def handle_target_add_medium_depressed(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量抑郁
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
    target_data.status.setdefault(14,0)
    now_value = target_data.status[14]
    now_add_value = (1 + now_value / 50) * add_time
    target_data.status[14] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(14,0)
    target_change.status[14] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_DEPRESSED)
def handle_target_add_large_depressed(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量抑郁
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
    target_data.status.setdefault(14,0)
    now_value = target_data.status[14]
    now_add_value = (1 + now_value / 10) * add_time
    target_data.status[14] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(14,0)
    target_change.status[14] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_SMALL_DEPRESSED)
def handle_target_sub_small_depressed(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少少量抑郁
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
    target_data.status.setdefault(14,0)
    now_value = target_data.status[14]
    now_sub_value = (1 + now_value / 100) * add_time
    target_data.status[14] -= now_sub_value
    target_data.status[14] = max(target_data.status[14],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(14,0)
    target_change.status[14] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_MEDIUM_DEPRESSED)
def handle_target_sub_medium_depressed(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少中量抑郁
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
    target_data.status.setdefault(14,0)
    now_value = target_data.status[14]
    now_sub_value = (1 + now_value / 50) * add_time
    target_data.status[14] -= now_sub_value
    target_data.status[14] = max(target_data.status[14],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(14,0)
    target_change.status[14] += now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_LARGE_DEPRESSED)
def handle_target_sub_large_depressed(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少大量抑郁
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
    target_data.status.setdefault(14,0)
    now_value = target_data.status[14]
    now_sub_value = (1 + now_value / 10) * add_time
    target_data.status[14] -= now_sub_value
    target_data.status[14] = max(target_data.status[14],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(14,0)
    target_change.status[14] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_ARROGANT)
def handle_target_add_small_arrogant(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量傲慢
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
    target_data.status.setdefault(15,0)
    now_value = target_data.status[15]
    now_add_value = (1 + now_value / 100) * add_time
    target_data.status[15] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(15,0)
    target_change.status[15] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_ARROGANT)
def handle_target_add_medium_arrogant(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量傲慢
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
    target_data.status.setdefault(15,0)
    now_value = target_data.status[15]
    now_add_value = (1 + now_value / 50) * add_time
    target_data.status[15] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(15,0)
    target_change.status[15] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_ARROGANT)
def handle_target_add_large_arrogant(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量傲慢
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
    target_data.status.setdefault(15,0)
    now_value = target_data.status[15]
    now_add_value = (1 + now_value / 10) * add_time
    target_data.status[15] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(15,0)
    target_change.status[15] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_SMALL_ARROGANT)
def handle_target_sub_small_arrogant(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少少量傲慢
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
    target_data.status.setdefault(15,0)
    now_value = target_data.status[15]
    now_sub_value = (1 + now_value / 100) * add_time
    target_data.status[15] -= now_sub_value
    target_data.status[15] = max(target_data.status[15],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(15,0)
    target_change.status[15] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_MEDIUM_ARROGANT)
def handle_target_sub_medium_arrogant(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少中量傲慢
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
    target_data.status.setdefault(15,0)
    now_value = target_data.status[15]
    now_sub_value = (1 + now_value / 50) * add_time
    target_data.status[15] -= now_sub_value
    target_data.status[15] = max(target_data.status[15],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(15,0)
    target_change.status[15] += now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_LARGE_ARROGANT)
def handle_target_sub_large_arrogant(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少大量傲慢
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
    target_data.status.setdefault(15,0)
    now_value = target_data.status[15]
    now_sub_value = (1 + now_value / 10) * add_time
    target_data.status[15] -= now_sub_value
    target_data.status[15] = max(target_data.status[15],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(15,0)
    target_change.status[15] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_ENVY)
def handle_target_add_small_envy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量嫉妒
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
    target_data.status.setdefault(16,0)
    now_value = target_data.status[16]
    now_add_value = (1 + now_value / 100) * add_time
    target_data.status[16] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(16,0)
    target_change.status[16] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_ENVY)
def handle_target_add_medium_envy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量嫉妒
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
    target_data.status.setdefault(16,0)
    now_value = target_data.status[16]
    now_add_value = (1 + now_value / 50) * add_time
    target_data.status[16] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(16,0)
    target_change.status[16] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_ENVY)
def handle_target_add_large_envy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量嫉妒
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
    target_data.status.setdefault(16,0)
    now_value = target_data.status[16]
    now_add_value = (1 + now_value / 10) * add_time
    target_data.status[16] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(16,0)
    target_change.status[16] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_SMALL_ENVY)
def handle_target_sub_small_envy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少少量嫉妒
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
    target_data.status.setdefault(16,0)
    now_value = target_data.status[16]
    now_sub_value = (1 + now_value / 100) * add_time
    target_data.status[16] -= now_sub_value
    target_data.status[16] = max(target_data.status[16],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(16,0)
    target_change.status[16] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_MEDIUM_ENVY)
def handle_target_sub_medium_envy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少中量嫉妒
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
    target_data.status.setdefault(16,0)
    now_value = target_data.status[16]
    now_sub_value = (1 + now_value / 50) * add_time
    target_data.status[16] -= now_sub_value
    target_data.status[16] = max(target_data.status[16],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(16,0)
    target_change.status[16] += now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_LARGE_ENVY)
def handle_target_sub_large_envy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少大量嫉妒
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
    target_data.status.setdefault(16,0)
    now_value = target_data.status[16]
    now_sub_value = (1 + now_value / 10) * add_time
    target_data.status[16] -= now_sub_value
    target_data.status[16] = max(target_data.status[16],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(16,0)
    target_change.status[16] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_RAGE)
def handle_target_add_small_rage(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量暴怒
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
    target_data.status.setdefault(17,0)
    now_value = target_data.status[17]
    now_add_value = (1 + now_value / 100) * add_time
    target_data.status[17] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(17,0)
    target_change.status[17] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_RAGE)
def handle_target_add_medium_rage(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量暴怒
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
    target_data.status.setdefault(17,0)
    now_value = target_data.status[17]
    now_add_value = (1 + now_value / 50) * add_time
    target_data.status[17] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(17,0)
    target_change.status[17] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_RAGE)
def handle_target_add_large_rage(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量暴怒
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
    target_data.status.setdefault(17,0)
    now_value = target_data.status[17]
    now_add_value = (1 + now_value / 10) * add_time
    target_data.status[17] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(17,0)
    target_change.status[17] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_SMALL_RAGE)
def handle_target_sub_small_rage(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少少量暴怒
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
    target_data.status.setdefault(17,0)
    now_value = target_data.status[17]
    now_sub_value = (1 + now_value / 100) * add_time
    target_data.status[17] -= now_sub_value
    target_data.status[17] = max(target_data.status[17],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(17,0)
    target_change.status[17] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_MEDIUM_RAGE)
def handle_target_sub_medium_rage(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少中量暴怒
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
    target_data.status.setdefault(17,0)
    now_value = target_data.status[17]
    now_sub_value = (1 + now_value / 50) * add_time
    target_data.status[17] -= now_sub_value
    target_data.status[17] = max(target_data.status[17],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(17,0)
    target_change.status[17] += now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_LARGE_RAGE)
def handle_target_sub_large_rage(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少大量暴怒
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
    target_data.status.setdefault(17,0)
    now_value = target_data.status[17]
    now_sub_value = (1 + now_value / 10) * add_time
    target_data.status[17] -= now_sub_value
    target_data.status[17] = max(target_data.status[17],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(17,0)
    target_change.status[17] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_LAZY)
def handle_target_add_small_lazy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量懒惰
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
    target_data.status.setdefault(18,0)
    now_value = target_data.status[18]
    now_add_value = (1 + now_value / 100) * add_time
    target_data.status[18] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(18,0)
    target_change.status[18] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_LAZY)
def handle_target_add_medium_lazy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量懒惰
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
    target_data.status.setdefault(18,0)
    now_value = target_data.status[18]
    now_add_value = (1 + now_value / 50) * add_time
    target_data.status[18] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(18,0)
    target_change.status[18] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_LAZY)
def handle_target_add_large_lazy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量懒惰
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
    target_data.status.setdefault(18,0)
    now_value = target_data.status[18]
    now_add_value = (1 + now_value / 10) * add_time
    target_data.status[18] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(18,0)
    target_change.status[18] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_SMALL_LAZY)
def handle_target_sub_small_lazy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少少量懒惰
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
    target_data.status.setdefault(18,0)
    now_value = target_data.status[18]
    now_sub_value = (1 + now_value / 100) * add_time
    target_data.status[18] -= now_sub_value
    target_data.status[18] = max(target_data.status[18],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(18,0)
    target_change.status[18] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_MEDIUM_LAZY)
def handle_target_sub_medium_lazy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少中量懒惰
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
    target_data.status.setdefault(18,0)
    now_value = target_data.status[18]
    now_sub_value = (1 + now_value / 50) * add_time
    target_data.status[18] -= now_sub_value
    target_data.status[18] = max(target_data.status[18],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(18,0)
    target_change.status[18] += now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_LARGE_LAZY)
def handle_target_sub_large_lazy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少大量懒惰
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
    target_data.status.setdefault(18,0)
    now_value = target_data.status[18]
    now_sub_value = (1 + now_value / 10) * add_time
    target_data.status[18] -= now_sub_value
    target_data.status[18] = max(target_data.status[18],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(18,0)
    target_change.status[18] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_GREEDY)
def handle_target_add_small_greedy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量贪婪
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
    target_data.status.setdefault(19,0)
    now_value = target_data.status[19]
    now_add_value = (1 + now_value / 100) * add_time
    target_data.status[19] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(19,0)
    target_change.status[19] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_GREEDY)
def handle_target_add_medium_greedy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量贪婪
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
    target_data.status.setdefault(19,0)
    now_value = target_data.status[19]
    now_add_value = (1 + now_value / 50) * add_time
    target_data.status[19] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(19,0)
    target_change.status[19] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_GREEDY)
def handle_target_add_large_greedy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量贪婪
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
    target_data.status.setdefault(19,0)
    now_value = target_data.status[19]
    now_add_value = (1 + now_value / 10) * add_time
    target_data.status[19] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(19,0)
    target_change.status[19] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_SMALL_GREEDY)
def handle_target_sub_small_greedy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少少量贪婪
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
    target_data.status.setdefault(19,0)
    now_value = target_data.status[19]
    now_sub_value = (1 + now_value / 100) * add_time
    target_data.status[19] -= now_sub_value
    target_data.status[19] = max(target_data.status[19],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(19,0)
    target_change.status[19] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_MEDIUM_GREEDY)
def handle_target_sub_medium_greedy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少中量贪婪
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
    target_data.status.setdefault(19,0)
    now_value = target_data.status[19]
    now_sub_value = (1 + now_value / 50) * add_time
    target_data.status[19] -= now_sub_value
    target_data.status[19] = max(target_data.status[19],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(19,0)
    target_change.status[19] += now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_LARGE_GREEDY)
def handle_target_sub_large_greedy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少大量贪婪
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
    target_data.status.setdefault(19,0)
    now_value = target_data.status[19]
    now_sub_value = (1 + now_value / 10) * add_time
    target_data.status[19] -= now_sub_value
    target_data.status[19] = max(target_data.status[19],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(19,0)
    target_change.status[19] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_GLUTTONY)
def handle_target_add_small_gluttony(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量暴食
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
    target_data.status.setdefault(20,0)
    now_value = target_data.status[20]
    now_add_value = (1 + now_value / 100) * add_time
    target_data.status[20] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(20,0)
    target_change.status[20] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_GLUTTONY)
def handle_target_add_medium_gluttony(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量暴食
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
    target_data.status.setdefault(20,0)
    now_value = target_data.status[20]
    now_add_value = (1 + now_value / 50) * add_time
    target_data.status[20] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(20,0)
    target_change.status[20] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_GLUTTONY)
def handle_target_add_large_gluttony(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量暴食
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
    target_data.status.setdefault(20,0)
    now_value = target_data.status[20]
    now_add_value = (1 + now_value / 10) * add_time
    target_data.status[20] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(20,0)
    target_change.status[20] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_SMALL_GLUTTONY)
def handle_target_sub_small_gluttony(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少少量暴食
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
    target_data.status.setdefault(20,0)
    now_value = target_data.status[20]
    now_sub_value = (1 + now_value / 100) * add_time
    target_data.status[20] -= now_sub_value
    target_data.status[20] = max(target_data.status[20],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(20,0)
    target_change.status[20] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_MEDIUM_GLUTTONY)
def handle_target_sub_medium_gluttony(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少中量暴食
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
    target_data.status.setdefault(20,0)
    now_value = target_data.status[20]
    now_sub_value = (1 + now_value / 50) * add_time
    target_data.status[20] -= now_sub_value
    target_data.status[20] = max(target_data.status[20],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(20,0)
    target_change.status[20] += now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_LARGE_GLUTTONY)
def handle_target_sub_large_gluttony(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少大量暴食
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
    target_data.status.setdefault(20,0)
    now_value = target_data.status[20]
    now_sub_value = (1 + now_value / 10) * add_time
    target_data.status[20] -= now_sub_value
    target_data.status[20] = max(target_data.status[20],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(20,0)
    target_change.status[20] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_LUST)
def handle_target_add_small_lust(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量色欲
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
    target_data.status.setdefault(21,0)
    now_value = target_data.status[21]
    now_add_value = (1 + now_value / 100) * add_time
    target_data.status[21] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(21,0)
    target_change.status[21] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_LUST)
def handle_target_add_medium_lust(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量色欲
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
    target_data.status.setdefault(21,0)
    now_value = target_data.status[21]
    now_add_value = (1 + now_value / 50) * add_time
    target_data.status[21] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(21,0)
    target_change.status[21] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_LUST)
def handle_target_add_large_lust(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量色欲
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
    target_data.status.setdefault(21,0)
    now_value = target_data.status[21]
    now_add_value = (1 + now_value / 10) * add_time
    target_data.status[21] += now_add_value
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(21,0)
    target_change.status[21] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_SMALL_LUST)
def handle_target_sub_small_lust(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少少量色欲
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
    target_data.status.setdefault(21,0)
    now_value = target_data.status[21]
    now_sub_value = (1 + now_value / 100) * add_time
    target_data.status[21] -= now_sub_value
    target_data.status[21] = max(target_data.status[21],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(21,0)
    target_change.status[21] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_MEDIUM_LUST)
def handle_target_sub_medium_lust(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少中量色欲
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
    target_data.status.setdefault(21,0)
    now_value = target_data.status[21]
    now_sub_value = (1 + now_value / 50) * add_time
    target_data.status[21] -= now_sub_value
    target_data.status[21] = max(target_data.status[21],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(21,0)
    target_change.status[21] += now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_LARGE_LUST)
def handle_target_sub_large_lust(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少大量色欲
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
    target_data.status.setdefault(21,0)
    now_value = target_data.status[21]
    now_sub_value = (1 + now_value / 10) * add_time
    target_data.status[21] -= now_sub_value
    target_data.status[21] = max(target_data.status[21],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(21,0)
    target_change.status[21] -= now_sub_value
