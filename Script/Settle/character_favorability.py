from Script.Design import settle_behavior, character, character_handle
from Script.Core import cache_control, constant, game_type


cache: game_type.Cache = cache_control.cache


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_FAVORABILITY)
def handle_add_small_favorability(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int
):
    """
    增加对交互对象的少量好感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    add_favorability = character.calculation_favorability(character_id, character_data.target_character_id, add_time)
    if character_data.target_character_id in character_data.social_contact_data:
        social = character_data.social_contact_data[character_data.target_character_id]
        if social:
            add_favorability *= social
    character_handle.add_favorability(character_data.target_character_id, character_id, add_favorability, None, now_time)


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_FAVORABILITY)
def handle_add_medium_favorability(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int
):
    """
    增加对交互对象的中量好感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    add_favorability = character.calculation_favorability(character_id, character_data.target_character_id, add_time * 5)
    if character_data.target_character_id in character_data.social_contact_data:
        social = character_data.social_contact_data[character_data.target_character_id]
        if social:
            add_favorability *= social
    character_handle.add_favorability(character_data.target_character_id, character_id, add_favorability, None, now_time)


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_FAVORABILITY)
def handle_add_large_favorability(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int
):
    """
    增加对交互对象的大量好感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    add_favorability = character.calculation_favorability(character_id, character_data.target_character_id, add_time * 10)
    if character_data.target_character_id in character_data.social_contact_data:
        social = character_data.social_contact_data[character_data.target_character_id]
        if social:
            add_favorability *= social
    character_handle.add_favorability(character_data.target_character_id, character_id, add_favorability, None, now_time)


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_SMALL_FAVORABILITY)
def handle_sub_small_favorability(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int
):
    """
    减少对交互对象的少量好感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    sub_favorability = add_time
    if character_data.target_character_id in character_data.social_contact_data:
        social = character_data.social_contact_data[character_data.target_character_id]
        if social:
            add_favorability /= social
    character_handle.add_favorability(character_data.target_character_id, character_id, -sub_favorability, None, now_time)


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_MEDIUM_FAVORABILITY)
def handle_sub_medium_favorability(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int
):
    """
    减少对交互对象的中量好感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    sub_favorability = add_time * 5
    if character_data.target_character_id in character_data.social_contact_data:
        social = character_data.social_contact_data[character_data.target_character_id]
        if social:
            add_favorability /= social
    character_handle.add_favorability(character_data.target_character_id, character_id, -sub_favorability, None, now_time)


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_LARGE_FAVORABILITY)
def handle_sub_large_favorability(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int
):
    """
    减少对交互对象的大量好感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    sub_favorability = add_time * 10
    if character_data.target_character_id in character_data.social_contact_data:
        social = character_data.social_contact_data[character_data.target_character_id]
        if social:
            add_favorability /= social
    character_handle.add_favorability(character_data.target_character_id, character_id, -sub_favorability, None, now_time)


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_FAVORABILITY)
def handle_target_add_small_favorability(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int
):
    """
    增加交互对象对角色的少量好感
    Keyword arguments:
    character_id -- 角色id
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
    add_favorability = character.calculation_favorability(character_id, character_data.target_character_id, add_time)
    if character_id in target_data.social_contact_data:
        social = target_data.social_contact_data[character_id]
        if social:
            add_favorability *= social
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change = change_data.target_change[target_data.cid]
    character_handle.add_favorability(character_id, target_data.cid, add_favorability, target_change, now_time)


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_FAVORABILITY)
def handle_target_add_medium_favorability(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int
):
    """
    增加交互对象对角色的中量好感
    Keyword arguments:
    character_id -- 角色id
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
    add_favorability = character.calculation_favorability(character_id, character_data.target_character_id, add_time * 5)
    if character_id in target_data.social_contact_data:
        social = target_data.social_contact_data[character_id]
        if social:
            add_favorability *= social
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change = change_data.target_change[target_data.cid]
    character_handle.add_favorability(character_id, target_data.cid, add_favorability, target_change, now_time)


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_FAVORABILITY)
def handle_target_add_large_favorability(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int
):
    """
    增加交互对象对角色的大量好感
    Keyword arguments:
    character_id -- 角色id
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
    add_favorability = character.calculation_favorability(character_id, character_data.target_character_id, add_time * 10)
    if character_id in target_data.social_contact_data:
        social = target_data.social_contact_data[character_id]
        if social:
            add_favorability *= social
    change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
    target_change = change_data.target_change[target_data.cid]
    character_handle.add_favorability(character_id, target_data.cid, add_favorability, target_change, now_time)


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_SMALL_FAVORABILITY)
def handle_target_sub_small_favorability(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int
):
    """
    减少对交互对象的少量好感
    Keyword arguments:
    character_id -- 角色id
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
    sub_favorability = add_time
    if character_id in target_data.social_contact_data:
        social = target_data.social_contact_data[character_id]
        if social:
            sub_favorability /= social
    character_handle.add_favorability(character_id,target_data.cid, -sub_favorability, None, now_time)


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_MEDIUM_FAVORABILITY)
def handle_target_sub_medium_favorability(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int
):
    """
    减少对交互对象的中量好感
    Keyword arguments:
    character_id -- 角色id
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
    sub_favorability = add_time * 5
    if character_id in target_data.social_contact_data:
        social = target_data.social_contact_data[character_id]
        if social:
            sub_favorability /= social
    character_handle.add_favorability(character_id,target_data.cid, -sub_favorability, None, now_time)


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_SUB_MEDIUM_FAVORABILITY)
def handle_target_sub_medium_favorability(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int
):
    """
    减少对交互对象的大量好感
    Keyword arguments:
    character_id -- 角色id
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
    sub_favorability = add_time * 10
    if character_id in target_data.social_contact_data:
        social = target_data.social_contact_data[character_id]
        if social:
            sub_favorability /= social
    character_handle.add_favorability(character_id,target_data.cid, -sub_favorability, None, now_time)
