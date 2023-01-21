from Script.Design import settle_behavior, constant
from Script.Core import cache_control, game_type


cache: game_type.Cache = cache_control.cache


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_APOTHECARY_EXPERIENCE)
def handle_add_small_apothecary(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量炼金学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(55,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[55]
    character_data.knowledge[55] += experience
    change_data.knowledge.setdefault(55,0)
    change_data.knowledge[55] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_APOTHECARY_EXPERIENCE)
def handle_add_medium_apothecary(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量炼金学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(55,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[55]
    character_data.knowledge[55] += experience
    change_data.knowledge.setdefault(55,0)
    change_data.knowledge[55] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_APOTHECARY_EXPERIENCE)
def handle_add_large_apothecary(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量炼金学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(55,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[55]
    character_data.knowledge[55] += experience
    change_data.knowledge.setdefault(55,0)
    change_data.knowledge[55] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_PARAPSYCHOLOGIES_EXPERIENCE)
def handle_add_small_parapsychologies(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量通灵学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(56,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[56]
    character_data.knowledge[56] += experience
    change_data.knowledge.setdefault(56,0)
    change_data.knowledge[56] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_PARAPSYCHOLOGIES_EXPERIENCE)
def handle_add_medium_parapsychologies(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量通灵学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(56,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[56]
    character_data.knowledge[56] += experience
    change_data.knowledge.setdefault(56,0)
    change_data.knowledge[56] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_PARAPSYCHOLOGIES_EXPERIENCE)
def handle_add_large_parapsychologies(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量通灵学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(56,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[56]
    character_data.knowledge[56] += experience
    change_data.knowledge.setdefault(56,0)
    change_data.knowledge[56] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_NUMEROLOGY_EXPERIENCE)
def handle_add_small_numerology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量灵数学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(57,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[57]
    character_data.knowledge[57] += experience
    change_data.knowledge.setdefault(57,0)
    change_data.knowledge[57] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_NUMEROLOGY_EXPERIENCE)
def handle_add_medium_numerology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量灵数学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(57,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[57]
    character_data.knowledge[57] += experience
    change_data.knowledge.setdefault(57,0)
    change_data.knowledge[57] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_NUMEROLOGY_EXPERIENCE)
def handle_add_large_numerology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量灵数学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(57,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[57]
    character_data.knowledge[57] += experience
    change_data.knowledge.setdefault(57,0)
    change_data.knowledge[57] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_PRACTISE_DIVINATION_EXPERIENCE)
def handle_add_small_practise_divination(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量占卜学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(58,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[58]
    character_data.knowledge[58] += experience
    change_data.knowledge.setdefault(58,0)
    change_data.knowledge[58] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_PRACTISE_DIVINATION_EXPERIENCE)
def handle_add_medium_practise_divination(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量占卜学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(58,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[58]
    character_data.knowledge[58] += experience
    change_data.knowledge.setdefault(58,0)
    change_data.knowledge[58] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_PRACTISE_DIVINATION_EXPERIENCE)
def handle_add_large_practise_divination(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量占卜学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(58,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[58]
    character_data.knowledge[58] += experience
    change_data.knowledge.setdefault(58,0)
    change_data.knowledge[58] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_PROPHECY_EXPERIENCE)
def handle_add_small_prophecy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量预言学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(59,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[59]
    character_data.knowledge[59] += experience
    change_data.knowledge.setdefault(59,0)
    change_data.knowledge[59] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_PROPHECY_EXPERIENCE)
def handle_add_medium_prophecy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量预言学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(59,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[59]
    character_data.knowledge[59] += experience
    change_data.knowledge.setdefault(59,0)
    change_data.knowledge[59] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_PROPHECY_EXPERIENCE)
def handle_add_large_prophecy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量预言学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(59,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[59]
    character_data.knowledge[59] += experience
    change_data.knowledge.setdefault(59,0)
    change_data.knowledge[59] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_ASTROLOGY_EXPERIENCE)
def handle_add_small_astrology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量占星学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(60,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[60]
    character_data.knowledge[60] += experience
    change_data.knowledge.setdefault(60,0)
    change_data.knowledge[60] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_ASTROLOGY_EXPERIENCE)
def handle_add_medium_astrology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量占星学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(60,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[60]
    character_data.knowledge[60] += experience
    change_data.knowledge.setdefault(60,0)
    change_data.knowledge[60] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_ASTROLOGY_EXPERIENCE)
def handle_add_large_astrology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量占星学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(60,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[60]
    character_data.knowledge[60] += experience
    change_data.knowledge.setdefault(60,0)
    change_data.knowledge[60] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_DEMONOLOGY_EXPERIENCE)
def handle_add_small_demonology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量恶魔学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(61,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[61]
    character_data.knowledge[61] += experience
    change_data.knowledge.setdefault(61,0)
    change_data.knowledge[61] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_DEMONOLOGY_EXPERIENCE)
def handle_add_medium_demonology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量恶魔学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(61,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[61]
    character_data.knowledge[61] += experience
    change_data.knowledge.setdefault(61,0)
    change_data.knowledge[61] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_DEMONOLOGY_EXPERIENCE)
def handle_add_large_demonology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量恶魔学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(61,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[61]
    character_data.knowledge[61] += experience
    change_data.knowledge.setdefault(61,0)
    change_data.knowledge[61] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_RITUAL_EXPERIENCE)
def handle_add_small_ritual(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量仪式学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(62,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[62]
    character_data.knowledge[62] += experience
    change_data.knowledge.setdefault(62,0)
    change_data.knowledge[62] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_RITUAL_EXPERIENCE)
def handle_add_medium_ritual(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量仪式学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(62,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[62]
    character_data.knowledge[62] += experience
    change_data.knowledge.setdefault(62,0)
    change_data.knowledge[62] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_RITUAL_EXPERIENCE)
def handle_add_large_ritual(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量仪式学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(62,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[62]
    character_data.knowledge[62] += experience
    change_data.knowledge.setdefault(62,0)
    change_data.knowledge[62] += experience
