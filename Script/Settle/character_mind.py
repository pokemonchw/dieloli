from Script.Design import settle_behavior, constant
from Script.Core import cache_control, game_type


cache: game_type.Cache = cache_control.cache


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_HAPPY)
def handle_add_small_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量快乐
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
    character_data.status.setdefault(8,0)
    now_value = character_data.status[8]
    now_add_value = (1 + now_value / 100) * add_time
    character_data.status[8] += now_add_value
    change_data.status.setdefault(8,0)
    change_data.status[8] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_HAPPY)
def handle_add_medium_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量快乐
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
    character_data.status.setdefault(8,0)
    now_value = character_data.status[8]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[8] += now_add_value
    change_data.status.setdefault(8,0)
    change_data.status[8] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_HAPPY)
def handle_add_large_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量快乐
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
    character_data.status.setdefault(8,0)
    now_value = character_data.status[8]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[8] += now_add_value
    change_data.status.setdefault(8,0)
    change_data.status[8] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_SMALL_HAPPY)
def handle_sub_small_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少少量快乐
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
    character_data.status.setdefault(8,0)
    if not character_data.status[8]:
        return
    now_value = character_data.status[8]
    now_sub_value = (1 + now_value / 100) * add_time
    character_data.status[8] -= now_sub_value
    character_data.status[8] = max(character_data.status[8],0)
    change_data.status.setdefault(8,0)
    change_data.status[8] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_MEDIUM_HAPPY)
def handle_sub_medium_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少中量快乐
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
    character_data.status.setdefault(8,0)
    if not character_data.status[8]:
        return
    now_value = character_data.status[8]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[8] -= now_sub_value
    character_data.status[8] = max(character_data.status[8],0)
    change_data.status.setdefault(8,0)
    change_data.status[8] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_LARGE_HAPPY)
def handle_sub_large_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少大量快乐
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
    character_data.status.setdefault(8,0)
    if not character_data.status[8]:
        return
    now_value = character_data.status[8]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[8] -= now_sub_value
    character_data.status[8] = max(character_data.status[8],0)
    change_data.status.setdefault(8,0)
    change_data.status[8] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_PAIN)
def handle_add_small_pain(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量痛苦
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
    character_data.status.setdefault(9,0)
    now_value = character_data.status[9]
    now_add_value = (1 + now_value / 100) * add_time
    character_data.status[9] += now_add_value
    change_data.status.setdefault(9,0)
    change_data.status[9] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_PAIN)
def handle_add_medium_pain(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量痛苦
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
    character_data.status.setdefault(9,0)
    now_value = character_data.status[9]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[9] += now_add_value
    change_data.status.setdefault(9,0)
    change_data.status[9] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_PAIN)
def handle_add_large_pain(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量痛苦
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
    character_data.status.setdefault(9,0)
    now_value = character_data.status[9]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[9] += now_add_value
    change_data.status.setdefault(9,0)
    change_data.status[9] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_SMALL_PAIN)
def handle_sub_small_pain(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少少量痛苦
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
    character_data.status.setdefault(9,0)
    if not character_data.status[9]:
        return
    now_value = character_data.status[9]
    now_sub_value = (1 + now_value / 100) * add_time
    character_data.status[9] -= now_sub_value
    character_data.status[9] = max(character_data.status[9],0)
    change_data.status.setdefault(9,0)
    change_data.status[9] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_MEDIUM_PAIN)
def handle_sub_medium_pain(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少中量痛苦
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
    character_data.status.setdefault(9,0)
    if not character_data.status[9]:
        return
    now_value = character_data.status[9]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[9] -= now_sub_value
    character_data.status[9] = max(character_data.status[9],0)
    change_data.status.setdefault(9,0)
    change_data.status[9] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_LARGE_PAIN)
def handle_sub_large_pain(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少大量痛苦
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
    character_data.status.setdefault(9,0)
    if not character_data.status[9]:
        return
    now_value = character_data.status[9]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[9] -= now_sub_value
    character_data.status[9] = max(character_data.status[9],0)
    change_data.status.setdefault(9,0)
    change_data.status[9] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_YEARN)
def handle_add_small_yearn(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量渴望
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
    character_data.status.setdefault(10,0)
    now_value = character_data.status[10]
    now_add_value = (1 + now_value / 100) * add_time
    character_data.status[10] += now_add_value
    change_data.status.setdefault(10,0)
    change_data.status[10] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_YEARN)
def handle_add_medium_yearn(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量渴望
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
    character_data.status.setdefault(10,0)
    now_value = character_data.status[10]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[10] += now_add_value
    change_data.status.setdefault(10,0)
    change_data.status[10] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_YEARN)
def handle_add_large_yearn(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量渴望
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
    character_data.status.setdefault(10,0)
    now_value = character_data.status[10]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[10] += now_add_value
    change_data.status.setdefault(10,0)
    change_data.status[10] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_SMALL_YEARN)
def handle_sub_small_yearn(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少少量渴望
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
    character_data.status.setdefault(10,0)
    if not character_data.status[10]:
        return
    now_value = character_data.status[10]
    now_sub_value = (1 + now_value / 100) * add_time
    character_data.status[10] -= now_sub_value
    character_data.status[10] = max(character_data.status[10],0)
    change_data.status.setdefault(10,0)
    change_data.status[10] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_MEDIUM_YEARN)
def handle_sub_medium_yearn(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少中量渴望
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
    character_data.status.setdefault(10,0)
    if not character_data.status[10]:
        return
    now_value = character_data.status[10]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[10] -= now_sub_value
    character_data.status[10] = max(character_data.status[10],0)
    change_data.status.setdefault(10,0)
    change_data.status[10] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_LARGE_YEARN)
def handle_sub_large_yearn(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少大量渴望
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
    character_data.status.setdefault(10,0)
    if not character_data.status[10]:
        return
    now_value = character_data.status[10]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[10] -= now_sub_value
    character_data.status[10] = max(character_data.status[10],0)
    change_data.status.setdefault(10,0)
    change_data.status[10] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_FEAR)
def handle_add_small_fear(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量恐惧
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
    character_data.status.setdefault(11,0)
    now_value = character_data.status[11]
    now_add_value = (1 + now_value / 100) * add_time
    character_data.status[11] += now_add_value
    change_data.status.setdefault(11,0)
    change_data.status[11] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_FEAR)
def handle_add_medium_fear(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量恐惧
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
    character_data.status.setdefault(11,0)
    now_value = character_data.status[11]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[11] += now_add_value
    change_data.status.setdefault(11,0)
    change_data.status[11] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_FEAR)
def handle_add_large_fear(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量恐惧
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
    character_data.status.setdefault(11,0)
    now_value = character_data.status[11]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[11] += now_add_value
    change_data.status.setdefault(11,0)
    change_data.status[11] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_SMALL_FEAR)
def handle_sub_small_fear(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少少量恐惧
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
    character_data.status.setdefault(11,0)
    if not character_data.status[11]:
        return
    now_value = character_data.status[11]
    now_sub_value = (1 + now_value / 100) * add_time
    character_data.status[11] -= now_sub_value
    character_data.status[11] = max(character_data.status[11],0)
    change_data.status.setdefault(11,0)
    change_data.status[11] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_MEDIUM_FEAR)
def handle_sub_medium_fear(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少中量恐惧
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
    character_data.status.setdefault(11,0)
    if not character_data.status[11]:
        return
    now_value = character_data.status[11]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[11] -= now_sub_value
    character_data.status[11] = max(character_data.status[11],0)
    change_data.status.setdefault(11,0)
    change_data.status[11] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_LARGE_FEAR)
def handle_sub_large_fear(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少大量恐惧
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
    character_data.status.setdefault(11,0)
    if not character_data.status[11]:
        return
    now_value = character_data.status[11]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[11] -= now_sub_value
    character_data.status[11] = max(character_data.status[11],0)
    change_data.status.setdefault(11,0)
    change_data.status[11] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_ANTIPATHY)
def handle_add_small_antipathy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量反感
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
    character_data.status.setdefault(12,0)
    now_value = character_data.status[12]
    now_add_value = (1 + now_value / 100) * add_time
    character_data.status[12] += now_add_value
    change_data.status.setdefault(12,0)
    change_data.status[12] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_ANTIPATHY)
def handle_add_medium_antipathy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量反感
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
    character_data.status.setdefault(12,0)
    now_value = character_data.status[12]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[12] += now_add_value
    change_data.status.setdefault(12,0)
    change_data.status[12] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_ANTIPATHY)
def handle_add_large_antipathy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量反感
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
    character_data.status.setdefault(12,0)
    now_value = character_data.status[12]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[12] += now_add_value
    change_data.status.setdefault(12,0)
    change_data.status[12] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_SMALL_ANTIPATHY)
def handle_sub_small_antipathy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少少量反感
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
    character_data.status.setdefault(12,0)
    if not character_data.status[12]:
        return
    now_value = character_data.status[12]
    now_sub_value = (1 + now_value / 100) * add_time
    character_data.status[12] -= now_sub_value
    character_data.status[12] = max(character_data.status[12],0)
    change_data.status.setdefault(12,0)
    change_data.status[12] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_MEDIUM_ANTIPATHY)
def handle_sub_medium_antipathy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少中量反感
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
    character_data.status.setdefault(12,0)
    if not character_data.status[12]:
        return
    now_value = character_data.status[12]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[12] -= now_sub_value
    character_data.status[12] = max(character_data.status[12],0)
    change_data.status.setdefault(12,0)
    change_data.status[12] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_LARGE_ANTIPATHY)
def handle_sub_large_antipathy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少大量反感
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
    character_data.status.setdefault(12,0)
    if not character_data.status[12]:
        return
    now_value = character_data.status[12]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[12] -= now_sub_value
    character_data.status[12] = max(character_data.status[12],0)
    change_data.status.setdefault(12,0)
    change_data.status[12] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_SHAME)
def handle_add_small_shame(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量羞耻
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
    character_data.status.setdefault(13,0)
    now_value = character_data.status[13]
    now_add_value = (1 + now_value / 100) * add_time
    character_data.status[13] += now_add_value
    change_data.status.setdefault(13,0)
    change_data.status[13] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_SHAME)
def handle_add_medium_shame(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量羞耻
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
    character_data.status.setdefault(13,0)
    now_value = character_data.status[13]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[13] += now_add_value
    change_data.status.setdefault(13,0)
    change_data.status[13] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_SHAME)
def handle_add_large_shame(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量羞耻
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
    character_data.status.setdefault(13,0)
    now_value = character_data.status[13]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[13] += now_add_value
    change_data.status.setdefault(13,0)
    change_data.status[13] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_SMALL_SHAME)
def handle_sub_small_shame(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少少量羞耻
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
    character_data.status.setdefault(13,0)
    if not character_data.status[13]:
        return
    now_value = character_data.status[13]
    now_sub_value = (1 + now_value / 100) * add_time
    character_data.status[13] -= now_sub_value
    character_data.status[13] = max(character_data.status[13],0)
    change_data.status.setdefault(13,0)
    change_data.status[13] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_MEDIUM_SHAME)
def handle_sub_medium_shame(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少中量羞耻
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
    character_data.status.setdefault(13,0)
    if not character_data.status[13]:
        return
    now_value = character_data.status[13]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[13] -= now_sub_value
    character_data.status[13] = max(character_data.status[13],0)
    change_data.status.setdefault(13,0)
    change_data.status[13] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_LARGE_SHAME)
def handle_sub_large_shame(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少大量羞耻
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
    character_data.status.setdefault(13,0)
    if not character_data.status[13]:
        return
    now_value = character_data.status[13]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[13] -= now_sub_value
    character_data.status[13] = max(character_data.status[13],0)
    change_data.status.setdefault(13,0)
    change_data.status[13] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_DEPRESSED)
def handle_add_small_depressed(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量抑郁
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
    character_data.status.setdefault(14,0)
    now_value = character_data.status[14]
    now_add_value = (1 + now_value / 100) * add_time
    character_data.status[14] += now_add_value
    change_data.status.setdefault(14,0)
    change_data.status[14] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_DEPRESSED)
def handle_add_medium_depressed(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量抑郁
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
    character_data.status.setdefault(14,0)
    now_value = character_data.status[14]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[14] += now_add_value
    change_data.status.setdefault(14,0)
    change_data.status[14] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_DEPRESSED)
def handle_add_large_depressed(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量抑郁
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
    character_data.status.setdefault(14,0)
    now_value = character_data.status[14]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[14] += now_add_value
    change_data.status.setdefault(14,0)
    change_data.status[14] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_SMALL_DEPRESSED)
def handle_sub_small_depressed(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少少量抑郁
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
    character_data.status.setdefault(14,0)
    if not character_data.status[14]:
        return
    now_value = character_data.status[14]
    now_sub_value = (1 + now_value / 100) * add_time
    character_data.status[14] -= now_sub_value
    character_data.status[14] = max(character_data.status[14],0)
    change_data.status.setdefault(14,0)
    change_data.status[14] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_MEDIUM_DEPRESSED)
def handle_sub_medium_depressed(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少中量抑郁
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
    character_data.status.setdefault(14,0)
    if not character_data.status[14]:
        return
    now_value = character_data.status[14]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[14] -= now_sub_value
    character_data.status[14] = max(character_data.status[14],0)
    change_data.status.setdefault(14,0)
    change_data.status[14] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_LARGE_DEPRESSED)
def handle_sub_large_depressed(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少大量抑郁
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
    character_data.status.setdefault(14,0)
    if not character_data.status[14]:
        return
    now_value = character_data.status[14]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[14] -= now_sub_value
    character_data.status[14] = max(character_data.status[14],0)
    change_data.status.setdefault(14,0)
    change_data.status[14] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_ARROGANT)
def handle_add_small_arrogant(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量傲慢
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
    character_data.status.setdefault(15,0)
    now_value = character_data.status[15]
    now_add_value = (1 + now_value / 100) * add_time
    character_data.status[15] += now_add_value
    change_data.status.setdefault(15,0)
    change_data.status[15] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_ARROGANT)
def handle_add_medium_arrogant(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量傲慢
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
    character_data.status.setdefault(15,0)
    now_value = character_data.status[15]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[15] += now_add_value
    change_data.status.setdefault(15,0)
    change_data.status[15] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_ARROGANT)
def handle_add_large_arrogant(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量傲慢
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
    character_data.status.setdefault(15,0)
    now_value = character_data.status[15]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[15] += now_add_value
    change_data.status.setdefault(15,0)
    change_data.status[15] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_SMALL_ARROGANT)
def handle_sub_small_arrogant(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少少量傲慢
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
    character_data.status.setdefault(15,0)
    if not character_data.status[15]:
        return
    now_value = character_data.status[15]
    now_sub_value = (1 + now_value / 100) * add_time
    character_data.status[15] -= now_sub_value
    character_data.status[15] = max(character_data.status[15],0)
    change_data.status.setdefault(15,0)
    change_data.status[15] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_MEDIUM_ARROGANT)
def handle_sub_medium_arrogant(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少中量傲慢
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
    character_data.status.setdefault(15,0)
    if not character_data.status[15]:
        return
    now_value = character_data.status[15]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[15] -= now_sub_value
    character_data.status[15] = max(character_data.status[15],0)
    change_data.status.setdefault(15,0)
    change_data.status[15] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_LARGE_ARROGANT)
def handle_sub_large_arrogant(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少大量傲慢
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
    character_data.status.setdefault(15,0)
    if not character_data.status[15]:
        return
    now_value = character_data.status[15]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[15] -= now_sub_value
    character_data.status[15] = max(character_data.status[15],0)
    change_data.status.setdefault(15,0)
    change_data.status[15] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_ENVY)
def handle_add_small_envy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量嫉妒
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
    character_data.status.setdefault(16,0)
    now_value = character_data.status[16]
    now_add_value = (1 + now_value / 100) * add_time
    character_data.status[16] += now_add_value
    change_data.status.setdefault(16,0)
    change_data.status[16] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_ENVY)
def handle_add_medium_envy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量嫉妒
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
    character_data.status.setdefault(16,0)
    now_value = character_data.status[16]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[16] += now_add_value
    change_data.status.setdefault(16,0)
    change_data.status[16] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_ENVY)
def handle_add_large_envy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量嫉妒
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
    character_data.status.setdefault(16,0)
    now_value = character_data.status[16]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[16] += now_add_value
    change_data.status.setdefault(16,0)
    change_data.status[16] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_SMALL_ENVY)
def handle_sub_small_envy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少少量嫉妒
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
    character_data.status.setdefault(16,0)
    if not character_data.status[16]:
        return
    now_value = character_data.status[16]
    now_sub_value = (1 + now_value / 100) * add_time
    character_data.status[16] -= now_sub_value
    character_data.status[16] = max(character_data.status[16],0)
    change_data.status.setdefault(16,0)
    change_data.status[16] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_MEDIUM_ENVY)
def handle_sub_medium_envy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少中量嫉妒
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
    character_data.status.setdefault(16,0)
    if not character_data.status[16]:
        return
    now_value = character_data.status[16]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[16] -= now_sub_value
    character_data.status[16] = max(character_data.status[16],0)
    change_data.status.setdefault(16,0)
    change_data.status[16] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_LARGE_ENVY)
def handle_sub_large_envy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少大量嫉妒
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
    character_data.status.setdefault(16,0)
    if not character_data.status[16]:
        return
    now_value = character_data.status[16]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[16] -= now_sub_value
    character_data.status[16] = max(character_data.status[16],0)
    change_data.status.setdefault(16,0)
    change_data.status[16] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_RAGE)
def handle_add_small_rage(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量暴怒
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
    character_data.status.setdefault(17,0)
    now_value = character_data.status[17]
    now_add_value = (1 + now_value / 100) * add_time
    character_data.status[17] += now_add_value
    change_data.status.setdefault(17,0)
    change_data.status[17] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_RAGE)
def handle_add_medium_rage(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量暴怒
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
    character_data.status.setdefault(17,0)
    now_value = character_data.status[17]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[17] += now_add_value
    change_data.status.setdefault(17,0)
    change_data.status[17] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_RAGE)
def handle_add_large_rage(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量暴怒
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
    character_data.status.setdefault(17,0)
    now_value = character_data.status[17]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[17] += now_add_value
    change_data.status.setdefault(17,0)
    change_data.status[17] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_SMALL_RAGE)
def handle_sub_small_rage(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少少量暴怒
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
    character_data.status.setdefault(17,0)
    if not character_data.status[17]:
        return
    now_value = character_data.status[17]
    now_sub_value = (1 + now_value / 100) * add_time
    character_data.status[17] -= now_sub_value
    character_data.status[17] = max(character_data.status[17],0)
    change_data.status.setdefault(17,0)
    change_data.status[17] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_MEDIUM_RAGE)
def handle_sub_medium_rage(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少中量暴怒
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
    character_data.status.setdefault(17,0)
    if not character_data.status[17]:
        return
    now_value = character_data.status[17]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[17] -= now_sub_value
    character_data.status[17] = max(character_data.status[17],0)
    change_data.status.setdefault(17,0)
    change_data.status[17] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_LARGE_RAGE)
def handle_sub_large_rage(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少大量暴怒
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
    character_data.status.setdefault(17,0)
    if not character_data.status[17]:
        return
    now_value = character_data.status[17]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[17] -= now_sub_value
    character_data.status[17] = max(character_data.status[17],0)
    change_data.status.setdefault(17,0)
    change_data.status[17] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_LAZY)
def handle_add_small_lazy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量懒惰
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
    character_data.status.setdefault(18,0)
    now_value = character_data.status[18]
    now_add_value = (1 + now_value / 100) * add_time
    character_data.status[18] += now_add_value
    change_data.status.setdefault(18,0)
    change_data.status[18] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_LAZY)
def handle_add_medium_lazy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量懒惰
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
    character_data.status.setdefault(18,0)
    now_value = character_data.status[18]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[18] += now_add_value
    change_data.status.setdefault(18,0)
    change_data.status[18] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_LAZY)
def handle_add_large_lazy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量懒惰
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
    character_data.status.setdefault(18,0)
    now_value = character_data.status[18]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[18] += now_add_value
    change_data.status.setdefault(18,0)
    change_data.status[18] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_SMALL_LAZY)
def handle_sub_small_lazy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少少量懒惰
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
    character_data.status.setdefault(18,0)
    if not character_data.status[18]:
        return
    now_value = character_data.status[18]
    now_sub_value = (1 + now_value / 100) * add_time
    character_data.status[18] -= now_sub_value
    character_data.status[18] = max(character_data.status[18],0)
    change_data.status.setdefault(18,0)
    change_data.status[18] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_MEDIUM_LAZY)
def handle_sub_medium_lazy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少中量懒惰
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
    character_data.status.setdefault(18,0)
    if not character_data.status[18]:
        return
    now_value = character_data.status[18]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[18] -= now_sub_value
    character_data.status[18] = max(character_data.status[18],0)
    change_data.status.setdefault(18,0)
    change_data.status[18] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_LARGE_LAZY)
def handle_sub_large_lazy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少大量懒惰
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
    character_data.status.setdefault(18,0)
    if not character_data.status[18]:
        return
    now_value = character_data.status[18]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[18] -= now_sub_value
    character_data.status[18] = max(character_data.status[18],0)
    change_data.status.setdefault(18,0)
    change_data.status[18] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_GREEDY)
def handle_add_small_greedy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量贪婪
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
    character_data.status.setdefault(19,0)
    now_value = character_data.status[19]
    now_add_value = (1 + now_value / 100) * add_time
    character_data.status[19] += now_add_value
    change_data.status.setdefault(19,0)
    change_data.status[19] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_GREEDY)
def handle_add_medium_greedy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量贪婪
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
    character_data.status.setdefault(19,0)
    now_value = character_data.status[19]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[19] += now_add_value
    change_data.status.setdefault(19,0)
    change_data.status[19] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_GREEDY)
def handle_add_large_greedy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量贪婪
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
    character_data.status.setdefault(19,0)
    now_value = character_data.status[19]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[19] += now_add_value
    change_data.status.setdefault(19,0)
    change_data.status[19] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_SMALL_GREEDY)
def handle_sub_small_greedy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少少量贪婪
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
    character_data.status.setdefault(19,0)
    if not character_data.status[19]:
        return
    now_value = character_data.status[19]
    now_sub_value = (1 + now_value / 100) * add_time
    character_data.status[19] -= now_sub_value
    character_data.status[19] = max(character_data.status[19],0)
    change_data.status.setdefault(19,0)
    change_data.status[19] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_MEDIUM_GREEDY)
def handle_sub_medium_greedy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少中量贪婪
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
    character_data.status.setdefault(19,0)
    if not character_data.status[19]:
        return
    now_value = character_data.status[19]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[19] -= now_sub_value
    character_data.status[19] = max(character_data.status[19],0)
    change_data.status.setdefault(19,0)
    change_data.status[19] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_LARGE_GREEDY)
def handle_sub_large_greedy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少大量贪婪
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
    character_data.status.setdefault(19,0)
    if not character_data.status[19]:
        return
    now_value = character_data.status[19]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[19] -= now_sub_value
    character_data.status[19] = max(character_data.status[19],0)
    change_data.status.setdefault(19,0)
    change_data.status[19] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_GLUTTONY)
def handle_add_small_gluttony(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量暴食
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
    character_data.status.setdefault(20,0)
    now_value = character_data.status[20]
    now_add_value = (1 + now_value / 100) * add_time
    character_data.status[20] += now_add_value
    change_data.status.setdefault(20,0)
    change_data.status[20] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_GLUTTONY)
def handle_add_medium_gluttony(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量暴食
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
    character_data.status.setdefault(20,0)
    now_value = character_data.status[20]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[20] += now_add_value
    change_data.status.setdefault(20,0)
    change_data.status[20] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_GLUTTONY)
def handle_add_large_gluttony(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量暴食
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
    character_data.status.setdefault(20,0)
    now_value = character_data.status[20]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[20] += now_add_value
    change_data.status.setdefault(20,0)
    change_data.status[20] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_SMALL_GLUTTONY)
def handle_sub_small_gluttony(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少少量暴食
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
    character_data.status.setdefault(20,0)
    if not character_data.status[20]:
        return
    now_value = character_data.status[20]
    now_sub_value = (1 + now_value / 100) * add_time
    character_data.status[20] -= now_sub_value
    character_data.status[20] = max(character_data.status[20],0)
    change_data.status.setdefault(20,0)
    change_data.status[20] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_MEDIUM_GLUTTONY)
def handle_sub_medium_gluttony(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少中量暴食
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
    character_data.status.setdefault(20,0)
    if not character_data.status[20]:
        return
    now_value = character_data.status[20]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[20] -= now_sub_value
    character_data.status[20] = max(character_data.status[20],0)
    change_data.status.setdefault(20,0)
    change_data.status[20] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_LARGE_GLUTTONY)
def handle_sub_large_gluttony(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少大量暴食
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
    character_data.status.setdefault(20,0)
    if not character_data.status[20]:
        return
    now_value = character_data.status[20]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[20] -= now_sub_value
    character_data.status[20] = max(character_data.status[20],0)
    change_data.status.setdefault(20,0)
    change_data.status[20] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_LUST)
def handle_add_small_lust(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量色欲
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
    character_data.status.setdefault(21,0)
    now_value = character_data.status[21]
    now_add_value = (1 + now_value / 100) * add_time
    character_data.status[21] += now_add_value
    change_data.status.setdefault(21,0)
    change_data.status[21] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_LUST)
def handle_add_medium_lust(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量色欲
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
    character_data.status.setdefault(21,0)
    now_value = character_data.status[21]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[21] += now_add_value
    change_data.status.setdefault(21,0)
    change_data.status[21] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_LUST)
def handle_add_large_lust(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量色欲
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
    character_data.status.setdefault(21,0)
    now_value = character_data.status[21]
    now_add_value = (1 + now_value / 50) * add_time
    character_data.status[21] += now_add_value
    change_data.status.setdefault(21,0)
    change_data.status[21] += now_add_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_SMALL_LUST)
def handle_sub_small_lust(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少少量色欲
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
    character_data.status.setdefault(21,0)
    if not character_data.status[21]:
        return
    now_value = character_data.status[21]
    now_sub_value = (1 + now_value / 100) * add_time
    character_data.status[21] -= now_sub_value
    character_data.status[21] = max(character_data.status[21],0)
    change_data.status.setdefault(21,0)
    change_data.status[21] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_MEDIUM_LUST)
def handle_sub_medium_lust(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少中量色欲
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
    character_data.status.setdefault(21,0)
    if not character_data.status[21]:
        return
    now_value = character_data.status[21]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[21] -= now_sub_value
    character_data.status[21] = max(character_data.status[21],0)
    change_data.status.setdefault(21,0)
    change_data.status[21] -= now_sub_value


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_LARGE_LUST)
def handle_sub_large_lust(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少大量色欲
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
    character_data.status.setdefault(21,0)
    if not character_data.status[21]:
        return
    now_value = character_data.status[21]
    now_sub_value = (1 + now_value / 50) * add_time
    character_data.status[21] -= now_sub_value
    character_data.status[21] = max(character_data.status[21],0)
    change_data.status.setdefault(21,0)
    change_data.status[21] -= now_sub_value
