from Script.Design import settle_behavior
from Script.Core import cache_control, constant, game_type


cache: game_type.Cache = cache_control.cache


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_MOUTH_EXPERIENCE)
def handle_add_small_mouth(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量嘴经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.sex_experience.setdefault(0,0)
    experience = add_time / 10
    character_data.sex_experience[0] += experience
    change_data.sex_experience.setdefault(0,0)
    change_data.sex_experience[0] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_MOUTH_EXPERIENCE)
def handle_add_medium_mouth(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量嘴经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.sex_experience.setdefault(0,0)
    experience = add_time / 5
    character_data.sex_experience[0] += experience
    change_data.sex_experience.setdefault(0,0)
    change_data.sex_experience[0] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_MOUTH_EXPERIENCE)
def handle_add_large_mouth(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量嘴经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.sex_experience.setdefault(0,0)
    experience = add_time
    character_data.sex_experience[0] += experience
    change_data.sex_experience.setdefault(0,0)
    change_data.sex_experience[0] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_CHEST_EXPERIENCE)
def handle_add_small_chest(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量胸经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.sex_experience.setdefault(1,0)
    experience = add_time / 10
    character_data.sex_experience[1] += experience
    change_data.sex_experience.setdefault(1,0)
    change_data.sex_experience[1] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_CHEST_EXPERIENCE)
def handle_add_medium_chest(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量胸经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.sex_experience.setdefault(1,0)
    experience = add_time / 5
    character_data.sex_experience[1] += experience
    change_data.sex_experience.setdefault(1,0)
    change_data.sex_experience[1] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_CHEST_EXPERIENCE)
def handle_add_large_chest(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量胸经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.sex_experience.setdefault(1,0)
    experience = add_time
    character_data.sex_experience[1] += experience
    change_data.sex_experience.setdefault(1,0)
    change_data.sex_experience[1] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_CLITORIS_EXPERIENCE)
def handle_add_small_clitoris(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量阴蒂经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.sex_experience.setdefault(2,0)
    experience = add_time / 10
    character_data.sex_experience[2] += experience
    change_data.sex_experience.setdefault(2,0)
    change_data.sex_experience[2] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_CLITORIS_EXPERIENCE)
def handle_add_medium_clitoris(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量阴蒂经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.sex_experience.setdefault(2,0)
    experience = add_time / 5
    character_data.sex_experience[2] += experience
    change_data.sex_experience.setdefault(2,0)
    change_data.sex_experience[2] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_CLITORIS_EXPERIENCE)
def handle_add_large_clitoris(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量阴蒂经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.sex_experience.setdefault(2,0)
    experience = add_time
    character_data.sex_experience[2] += experience
    change_data.sex_experience.setdefault(2,0)
    change_data.sex_experience[2] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_PENIS_EXPERIENCE)
def handle_add_small_penis(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量阴茎经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.sex_experience.setdefault(3,0)
    experience = add_time / 10
    character_data.sex_experience[3] += experience
    change_data.sex_experience.setdefault(3,0)
    change_data.sex_experience[3] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_PENIS_EXPERIENCE)
def handle_add_medium_penis(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量阴茎经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.sex_experience.setdefault(3,0)
    experience = add_time / 5
    character_data.sex_experience[3] += experience
    change_data.sex_experience.setdefault(3,0)
    change_data.sex_experience[3] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_PENIS_EXPERIENCE)
def handle_add_large_penis(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量阴茎经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.sex_experience.setdefault(3,0)
    experience = add_time
    character_data.sex_experience[3] += experience
    change_data.sex_experience.setdefault(3,0)
    change_data.sex_experience[3] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_VAGINA_EXPERIENCE)
def handle_add_small_vagina(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量阴道经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.sex_experience.setdefault(4,0)
    experience = add_time / 10
    character_data.sex_experience[4] += experience
    change_data.sex_experience.setdefault(4,0)
    change_data.sex_experience[4] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_VAGINA_EXPERIENCE)
def handle_add_medium_vagina(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量阴道经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.sex_experience.setdefault(4,0)
    experience = add_time / 5
    character_data.sex_experience[4] += experience
    change_data.sex_experience.setdefault(4,0)
    change_data.sex_experience[4] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_VAGINA_EXPERIENCE)
def handle_add_large_vagina(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量阴道经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.sex_experience.setdefault(4,0)
    experience = add_time
    character_data.sex_experience[4] += experience
    change_data.sex_experience.setdefault(4,0)
    change_data.sex_experience[4] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_ANUS_EXPERIENCE)
def handle_add_small_anus(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量肛门经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.sex_experience.setdefault(5,0)
    experience = add_time / 10
    character_data.sex_experience[5] += experience
    change_data.sex_experience.setdefault(5,0)
    change_data.sex_experience[5] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_ANUS_EXPERIENCE)
def handle_add_medium_anus(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量肛门经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.sex_experience.setdefault(5,0)
    experience = add_time / 5
    character_data.sex_experience[5] += experience
    change_data.sex_experience.setdefault(5,0)
    change_data.sex_experience[5] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_ANUS_EXPERIENCE)
def handle_add_large_anus(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量肛门经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.sex_experience.setdefault(5,0)
    experience = add_time
    character_data.sex_experience[5] += experience
    change_data.sex_experience.setdefault(5,0)
    change_data.sex_experience[5] += experience
