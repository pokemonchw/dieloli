from Script.Design import settle_behavior
from Script.Core import cache_control, constant, game_type


cache: game_type.Cache = cache_control.cache


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_MOUTH_EXPERIENCE)
def handle_target_add_small_mouth(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量嘴经验
    Keyword arguments:
    character_id -- 角色id
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
    target_data.sex_experience.setdefault(0,0)
    experience = add_time / 10
    target_data.sex_experience[0] += experience
    change_data.target_change.setdefault(target_data.cid,game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.sex_experience.setdefault(0,0)
    target_change.sex_experience[0] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_MOUTH_EXPERIENCE)
def handle_target_add_medium_mouth(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量嘴经验
    Keyword arguments:
    character_id -- 角色id
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
    target_data.sex_experience.setdefault(0,0)
    experience = add_time / 5
    target_data.sex_experience[0] += experience
    change_data.target_change.setdefault(target_data.cid,game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.sex_experience.setdefault(0,0)
    target_change.sex_experience[0] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_MOUTH_EXPERIENCE)
def handle_target_add_large_mouth(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量嘴经验
    Keyword arguments:
    character_id -- 角色id
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
    target_data.sex_experience.setdefault(0,0)
    experience = add_time
    target_data.sex_experience[0] += experience
    change_data.target_change.setdefault(target_data.cid,game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.sex_experience.setdefault(0,0)
    target_change.sex_experience[0] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_CHEST_EXPERIENCE)
def handle_target_add_small_chest(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量胸经验
    Keyword arguments:
    character_id -- 角色id
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
    target_data.sex_experience.setdefault(1,0)
    experience = add_time / 10
    target_data.sex_experience[1] += experience
    change_data.target_change.setdefault(target_data.cid,game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.sex_experience.setdefault(1,0)
    target_change.sex_experience[1] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_CHEST_EXPERIENCE)
def handle_target_add_medium_chest(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量胸经验
    Keyword arguments:
    character_id -- 角色id
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
    target_data.sex_experience.setdefault(1,0)
    experience = add_time / 5
    target_data.sex_experience[1] += experience
    change_data.target_change.setdefault(target_data.cid,game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.sex_experience.setdefault(1,0)
    target_change.sex_experience[1] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_CHEST_EXPERIENCE)
def handle_target_add_large_chest(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量胸经验
    Keyword arguments:
    character_id -- 角色id
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
    target_data.sex_experience.setdefault(1,0)
    experience = add_time
    target_data.sex_experience[1] += experience
    change_data.target_change.setdefault(target_data.cid,game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.sex_experience.setdefault(1,0)
    target_change.sex_experience[1] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_CLITORIS_EXPERIENCE)
def handle_target_add_small_clitoris(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量阴蒂经验
    Keyword arguments:
    character_id -- 角色id
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
    target_data.sex_experience.setdefault(2,0)
    experience = add_time / 10
    target_data.sex_experience[2] += experience
    change_data.target_change.setdefault(target_data.cid,game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.sex_experience.setdefault(2,0)
    target_change.sex_experience[2] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_CLITORIS_EXPERIENCE)
def handle_target_add_medium_clitoris(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量阴蒂经验
    Keyword arguments:
    character_id -- 角色id
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
    target_data.sex_experience.setdefault(2,0)
    experience = add_time / 5
    target_data.sex_experience[2] += experience
    change_data.target_change.setdefault(target_data.cid,game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.sex_experience.setdefault(2,0)
    target_change.sex_experience[2] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_CLITORIS_EXPERIENCE)
def handle_target_add_large_clitoris(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量阴蒂经验
    Keyword arguments:
    character_id -- 角色id
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
    target_data.sex_experience.setdefault(2,0)
    experience = add_time
    target_data.sex_experience[2] += experience
    change_data.target_change.setdefault(target_data.cid,game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.sex_experience.setdefault(2,0)
    target_change.sex_experience[2] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_PENIS_EXPERIENCE)
def handle_target_add_small_penis(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量阴茎经验
    Keyword arguments:
    character_id -- 角色id
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
    target_data.sex_experience.setdefault(3,0)
    experience = add_time / 10
    target_data.sex_experience[3] += experience
    change_data.target_change.setdefault(target_data.cid,game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.sex_experience.setdefault(3,0)
    target_change.sex_experience[3] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_PENIS_EXPERIENCE)
def handle_target_add_medium_penis(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量阴茎经验
    Keyword arguments:
    character_id -- 角色id
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
    target_data.sex_experience.setdefault(3,0)
    experience = add_time / 5
    target_data.sex_experience[3] += experience
    change_data.target_change.setdefault(target_data.cid,game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.sex_experience.setdefault(3,0)
    target_change.sex_experience[3] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_PENIS_EXPERIENCE)
def handle_target_add_large_penis(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量阴茎经验
    Keyword arguments:
    character_id -- 角色id
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
    target_data.sex_experience.setdefault(3,0)
    experience = add_time
    target_data.sex_experience[3] += experience
    change_data.target_change.setdefault(target_data.cid,game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.sex_experience.setdefault(3,0)
    target_change.sex_experience[3] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_VAGINA_EXPERIENCE)
def handle_target_add_small_vagina(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量阴道经验
    Keyword arguments:
    character_id -- 角色id
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
    target_data.sex_experience.setdefault(4,0)
    experience = add_time / 10
    target_data.sex_experience[4] += experience
    change_data.target_change.setdefault(target_data.cid,game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.sex_experience.setdefault(4,0)
    target_change.sex_experience[4] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_VAGINA_EXPERIENCE)
def handle_target_add_medium_vagina(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量阴道经验
    Keyword arguments:
    character_id -- 角色id
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
    target_data.sex_experience.setdefault(4,0)
    experience = add_time / 5
    target_data.sex_experience[4] += experience
    change_data.target_change.setdefault(target_data.cid,game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.sex_experience.setdefault(4,0)
    target_change.sex_experience[4] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_VAGINA_EXPERIENCE)
def handle_target_add_large_vagina(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量阴道经验
    Keyword arguments:
    character_id -- 角色id
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
    target_data.sex_experience.setdefault(4,0)
    experience = add_time
    target_data.sex_experience[4] += experience
    change_data.target_change.setdefault(target_data.cid,game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.sex_experience.setdefault(4,0)
    target_change.sex_experience[4] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_ANUS_EXPERIENCE)
def handle_target_add_small_anus(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量肛门经验
    Keyword arguments:
    character_id -- 角色id
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
    target_data.sex_experience.setdefault(5,0)
    experience = add_time / 10
    target_data.sex_experience[5] += experience
    change_data.target_change.setdefault(target_data.cid,game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.sex_experience.setdefault(5,0)
    target_change.sex_experience[5] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_ANUS_EXPERIENCE)
def handle_target_add_medium_anus(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量肛门经验
    Keyword arguments:
    character_id -- 角色id
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
    target_data.sex_experience.setdefault(5,0)
    experience = add_time / 5
    target_data.sex_experience[5] += experience
    change_data.target_change.setdefault(target_data.cid,game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.sex_experience.setdefault(5,0)
    target_change.sex_experience[5] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_ANUS_EXPERIENCE)
def handle_target_add_large_anus(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量肛门经验
    Keyword arguments:
    character_id -- 角色id
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
    target_data.sex_experience.setdefault(5,0)
    experience = add_time
    target_data.sex_experience[5] += experience
    change_data.target_change.setdefault(target_data.cid,game_type.TargetChange())
    target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
    target_change.sex_experience.setdefault(5,0)
    target_change.sex_experience[5] += experience
