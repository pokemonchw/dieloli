from Script.Design import settle_behavior, attr_calculation
from Script.Core import cache_control, constant, game_type


cache: game_type.Cache = cache_control.cache


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_MOUTH_HAPPY)
def handle_target_add_small_mouth_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量嘴部快感快感
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
    target_data.status.setdefault(0,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[0]) + target_data.status[0] / 100
    target_data.status[0] += target_happy
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(0,0)
    target_change.status[0] += target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_MOUTH_HAPPY)
def handle_target_add_medium_mouth_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量嘴部快感快感
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
    target_data.status.setdefault(0,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[0]) + target_data.status[0] / 50
    target_data.status[0] += target_happy
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(0,0)
    target_change.status[0] += target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_MOUTH_HAPPY)
def handle_target_add_large_mouth_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量嘴部快感快感
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
    target_data.status.setdefault(0,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[0]) + target_data.status[0] / 10
    target_data.status[0] += target_happy
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(0,0)
    target_change.status[0] += target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_SMALL_MOUTH_HAPPY)
def handle_target_sub_small_mouth_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少少量嘴部快感快感
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
    target_data.status.setdefault(0,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[0]) + target_data.status[0] / 100
    target_data.status[0] -= target_happy
    target_data.status[0] = max(target_data.status[0],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(0,0)
    target_change.status[0] -= target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_MEDIUM_MOUTH_HAPPY)
def handle_target_sub_medium_mouth_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少中量嘴部快感快感
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
    target_data.status.setdefault(0,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[0]) + target_data.status[0] / 50
    target_data.status[0] -= target_happy
    target_data.status[0] = max(target_data.status[0],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(0,0)
    target_change.status[0] -= target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_LARGE_MOUTH_HAPPY)
def handle_target_sub_large_mouth_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少大量嘴部快感快感
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
    target_data.status.setdefault(0,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[0]) + target_data.status[0] / 10
    target_data.status[0] -= target_happy
    target_data.status[0] = max(target_data.status[0],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(0,0)
    target_change.status[0] -= target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_CHEST_HAPPY)
def handle_target_add_small_chest_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量胸部快感快感
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
    target_data.status.setdefault(1,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[1]) + target_data.status[1] / 100
    target_data.status[1] += target_happy
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(1,0)
    target_change.status[1] += target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_CHEST_HAPPY)
def handle_target_add_medium_chest_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量胸部快感快感
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
    target_data.status.setdefault(1,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[1]) + target_data.status[1] / 50
    target_data.status[1] += target_happy
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(1,0)
    target_change.status[1] += target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_CHEST_HAPPY)
def handle_target_add_large_chest_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量胸部快感快感
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
    target_data.status.setdefault(1,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[1]) + target_data.status[1] / 10
    target_data.status[1] += target_happy
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(1,0)
    target_change.status[1] += target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_SMALL_CHEST_HAPPY)
def handle_target_sub_small_chest_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少少量胸部快感快感
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
    target_data.status.setdefault(1,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[1]) + target_data.status[1] / 100
    target_data.status[1] -= target_happy
    target_data.status[1] = max(target_data.status[1],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(1,0)
    target_change.status[1] -= target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_MEDIUM_CHEST_HAPPY)
def handle_target_sub_medium_chest_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少中量胸部快感快感
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
    target_data.status.setdefault(1,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[1]) + target_data.status[1] / 50
    target_data.status[1] -= target_happy
    target_data.status[1] = max(target_data.status[1],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(1,0)
    target_change.status[1] -= target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_LARGE_CHEST_HAPPY)
def handle_target_sub_large_chest_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少大量胸部快感快感
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
    target_data.status.setdefault(1,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[1]) + target_data.status[1] / 10
    target_data.status[1] -= target_happy
    target_data.status[1] = max(target_data.status[1],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(1,0)
    target_change.status[1] -= target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_VAGINA_HAPPY)
def handle_target_add_small_vagina_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量阴道快感快感
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
    target_data.status.setdefault(2,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[4]) + target_data.status[2] / 100
    target_data.status[2] += target_happy
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(2,0)
    target_change.status[2] += target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_VAGINA_HAPPY)
def handle_target_add_medium_vagina_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量阴道快感快感
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
    target_data.status.setdefault(2,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[4]) + target_data.status[2] / 50
    target_data.status[2] += target_happy
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(2,0)
    target_change.status[2] += target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_VAGINA_HAPPY)
def handle_target_add_large_vagina_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量阴道快感快感
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
    target_data.status.setdefault(2,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[4]) + target_data.status[2] / 10
    target_data.status[2] += target_happy
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(2,0)
    target_change.status[2] += target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_SMALL_VAGINA_HAPPY)
def handle_target_sub_small_vagina_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少少量阴道快感快感
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
    target_data.status.setdefault(2,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[4]) + target_data.status[2] / 100
    target_data.status[2] -= target_happy
    target_data.status[2] = max(target_data.status[2],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(2,0)
    target_change.status[2] -= target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_MEDIUM_VAGINA_HAPPY)
def handle_target_sub_medium_vagina_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少中量阴道快感快感
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
    target_data.status.setdefault(2,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[4]) + target_data.status[2] / 50
    target_data.status[2] -= target_happy
    target_data.status[2] = max(target_data.status[2],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(2,0)
    target_change.status[2] -= target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_LARGE_VAGINA_HAPPY)
def handle_target_sub_large_vagina_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少大量阴道快感快感
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
    target_data.status.setdefault(2,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[4]) + target_data.status[2] / 10
    target_data.status[2] -= target_happy
    target_data.status[2] = max(target_data.status[2],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(2,0)
    target_change.status[2] -= target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_CLITORIS_HAPPY)
def handle_target_add_small_clitoris_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量阴蒂快感快感
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
    target_data.status.setdefault(3,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[2]) + target_data.status[3] / 100
    target_data.status[3] += target_happy
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(3,0)
    target_change.status[3] += target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_CLITORIS_HAPPY)
def handle_target_add_medium_clitoris_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量阴蒂快感快感
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
    target_data.status.setdefault(3,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[2]) + target_data.status[3] / 50
    target_data.status[3] += target_happy
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(3,0)
    target_change.status[3] += target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_CLITORIS_HAPPY)
def handle_target_add_large_clitoris_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量阴蒂快感快感
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
    target_data.status.setdefault(3,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[2]) + target_data.status[3] / 10
    target_data.status[3] += target_happy
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(3,0)
    target_change.status[3] += target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_SMALL_CLITORIS_HAPPY)
def handle_target_sub_small_clitoris_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少少量阴蒂快感快感
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
    target_data.status.setdefault(3,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[2]) + target_data.status[3] / 100
    target_data.status[3] -= target_happy
    target_data.status[3] = max(target_data.status[3],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(3,0)
    target_change.status[3] -= target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_MEDIUM_CLITORIS_HAPPY)
def handle_target_sub_medium_clitoris_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少中量阴蒂快感快感
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
    target_data.status.setdefault(3,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[2]) + target_data.status[3] / 50
    target_data.status[3] -= target_happy
    target_data.status[3] = max(target_data.status[3],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(3,0)
    target_change.status[3] -= target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_LARGE_CLITORIS_HAPPY)
def handle_target_sub_large_clitoris_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少大量阴蒂快感快感
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
    target_data.status.setdefault(3,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[2]) + target_data.status[3] / 10
    target_data.status[3] -= target_happy
    target_data.status[3] = max(target_data.status[3],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(3,0)
    target_change.status[3] -= target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_ANUS_HAPPY)
def handle_target_add_small_anus_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量肛门快感快感
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
    target_data.status.setdefault(4,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[5]) + target_data.status[4] / 100
    target_data.status[4] += target_happy
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(4,0)
    target_change.status[4] += target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_ANUS_HAPPY)
def handle_target_add_medium_anus_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量肛门快感快感
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
    target_data.status.setdefault(4,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[5]) + target_data.status[4] / 50
    target_data.status[4] += target_happy
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(4,0)
    target_change.status[4] += target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_ANUS_HAPPY)
def handle_target_add_large_anus_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量肛门快感快感
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
    target_data.status.setdefault(4,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[5]) + target_data.status[4] / 10
    target_data.status[4] += target_happy
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(4,0)
    target_change.status[4] += target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_SMALL_ANUS_HAPPY)
def handle_target_sub_small_anus_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少少量肛门快感快感
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
    target_data.status.setdefault(4,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[5]) + target_data.status[4] / 100
    target_data.status[4] -= target_happy
    target_data.status[4] = max(target_data.status[4],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(4,0)
    target_change.status[4] -= target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_MEDIUM_ANUS_HAPPY)
def handle_target_sub_medium_anus_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少中量肛门快感快感
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
    target_data.status.setdefault(4,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[5]) + target_data.status[4] / 50
    target_data.status[4] -= target_happy
    target_data.status[4] = max(target_data.status[4],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(4,0)
    target_change.status[4] -= target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_LARGE_ANUS_HAPPY)
def handle_target_sub_large_anus_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少大量肛门快感快感
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
    target_data.status.setdefault(4,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[5]) + target_data.status[4] / 10
    target_data.status[4] -= target_happy
    target_data.status[4] = max(target_data.status[4],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(4,0)
    target_change.status[4] -= target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_PENIS_HAPPY)
def handle_target_add_small_penis_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量阴茎快感快感
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
    target_data.status.setdefault(5,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[3]) + target_data.status[5] / 100
    target_data.status[5] += target_happy
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(5,0)
    target_change.status[5] += target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_PENIS_HAPPY)
def handle_target_add_medium_penis_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量阴茎快感快感
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
    target_data.status.setdefault(5,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[3]) + target_data.status[5] / 50
    target_data.status[5] += target_happy
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(5,0)
    target_change.status[5] += target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_PENIS_HAPPY)
def handle_target_add_large_penis_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量阴茎快感快感
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
    target_data.status.setdefault(5,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[3]) + target_data.status[5] / 10
    target_data.status[5] += target_happy
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(5,0)
    target_change.status[5] += target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_SMALL_PENIS_HAPPY)
def handle_target_sub_small_penis_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少少量阴茎快感快感
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
    target_data.status.setdefault(5,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[3]) + target_data.status[5] / 100
    target_data.status[5] -= target_happy
    target_data.status[5] = max(target_data.status[5],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(5,0)
    target_change.status[5] -= target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_MEDIUM_PENIS_HAPPY)
def handle_target_sub_medium_penis_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少中量阴茎快感快感
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
    target_data.status.setdefault(5,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[3]) + target_data.status[5] / 50
    target_data.status[5] -= target_happy
    target_data.status[5] = max(target_data.status[5],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(5,0)
    target_change.status[5] -= target_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_LARGE_PENIS_HAPPY)
def handle_target_sub_large_penis_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象减少大量阴茎快感快感
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
    target_data.status.setdefault(5,0)
    target_happy = add_time
    target_happy *= 1 + attr_calculation.get_experience_level_weight(target_data.sex_experience[3]) + target_data.status[5] / 10
    target_data.status[5] -= target_happy
    target_data.status[5] = max(target_data.status[5],0)
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.status.setdefault(5,0)
    target_change.status[5] -= target_happy
