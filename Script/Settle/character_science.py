from Script.Design import settle_behavior, constant
from Script.Core import cache_control, game_type


cache: game_type.Cache = cache_control.cache


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_MECHANICS_EXPERIENCE)
def handle_add_small_mechanics(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量机械学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(32,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[32]
    character_data.knowledge[32] += experience
    change_data.knowledge.setdefault(32,0)
    change_data.knowledge[32] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_MECHANICS_EXPERIENCE)
def handle_add_medium_mechanics(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量机械学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(32,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[32]
    character_data.knowledge[32] += experience
    change_data.knowledge.setdefault(32,0)
    change_data.knowledge[32] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_MECHANICS_EXPERIENCE)
def handle_add_large_mechanics(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量机械学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(32,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[32]
    character_data.knowledge[32] += experience
    change_data.knowledge.setdefault(32,0)
    change_data.knowledge[32] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_ELECTRONICS_EXPERIENCE)
def handle_add_small_electronics(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量电子学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(33,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[33]
    character_data.knowledge[33] += experience
    change_data.knowledge.setdefault(33,0)
    change_data.knowledge[33] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_ELECTRONICS_EXPERIENCE)
def handle_add_medium_electronics(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量电子学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(33,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[33]
    character_data.knowledge[33] += experience
    change_data.knowledge.setdefault(33,0)
    change_data.knowledge[33] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_ELECTRONICS_EXPERIENCE)
def handle_add_large_electronics(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量电子学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(33,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[33]
    character_data.knowledge[33] += experience
    change_data.knowledge.setdefault(33,0)
    change_data.knowledge[33] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_COMPUTER_SCIENCE_EXPERIENCE)
def handle_add_small_computer_science(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量计算机学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(34,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[34]
    character_data.knowledge[34] += experience
    change_data.knowledge.setdefault(34,0)
    change_data.knowledge[34] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_COMPUTER_SCIENCE_EXPERIENCE)
def handle_add_medium_computer_science(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量计算机学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(34,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[34]
    character_data.knowledge[34] += experience
    change_data.knowledge.setdefault(34,0)
    change_data.knowledge[34] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_COMPUTER_SCIENCE_EXPERIENCE)
def handle_add_large_computer_science(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量计算机学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(34,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[34]
    character_data.knowledge[34] += experience
    change_data.knowledge.setdefault(34,0)
    change_data.knowledge[34] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_CRYPTOGRAPHY_EXPERIENCE)
def handle_add_small_cryptography(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量密码学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(35,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[35]
    character_data.knowledge[35] += experience
    change_data.knowledge.setdefault(35,0)
    change_data.knowledge[35] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_CRYPTOGRAPHY_EXPERIENCE)
def handle_add_medium_cryptography(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量密码学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(35,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[35]
    character_data.knowledge[35] += experience
    change_data.knowledge.setdefault(35,0)
    change_data.knowledge[35] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_CRYPTOGRAPHY_EXPERIENCE)
def handle_add_large_cryptography(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量密码学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(35,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[35]
    character_data.knowledge[35] += experience
    change_data.knowledge.setdefault(35,0)
    change_data.knowledge[35] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_CHEMISTRY_EXPERIENCE)
def handle_add_small_chemistry(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量化学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(36,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[36]
    character_data.knowledge[36] += experience
    change_data.knowledge.setdefault(36,0)
    change_data.knowledge[36] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_CHEMISTRY_EXPERIENCE)
def handle_add_medium_chemistry(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量化学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(36,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[36]
    character_data.knowledge[36] += experience
    change_data.knowledge.setdefault(36,0)
    change_data.knowledge[36] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_CHEMISTRY_EXPERIENCE)
def handle_add_large_chemistry(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量化学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(36,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[36]
    character_data.knowledge[36] += experience
    change_data.knowledge.setdefault(36,0)
    change_data.knowledge[36] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_BIOLOGY_EXPERIENCE)
def handle_add_small_biology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量生物学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(37,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[37]
    character_data.knowledge[37] += experience
    change_data.knowledge.setdefault(37,0)
    change_data.knowledge[37] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_BIOLOGY_EXPERIENCE)
def handle_add_medium_biology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量生物学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(37,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[37]
    character_data.knowledge[37] += experience
    change_data.knowledge.setdefault(37,0)
    change_data.knowledge[37] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_BIOLOGY_EXPERIENCE)
def handle_add_large_biology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量生物学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(37,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[37]
    character_data.knowledge[37] += experience
    change_data.knowledge.setdefault(37,0)
    change_data.knowledge[37] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_MATHEMATICS_EXPERIENCE)
def handle_add_small_mathematics(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量数学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(38,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[38]
    character_data.knowledge[38] += experience
    change_data.knowledge.setdefault(38,0)
    change_data.knowledge[38] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_MATHEMATICS_EXPERIENCE)
def handle_add_medium_mathematics(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量数学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(38,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[38]
    character_data.knowledge[38] += experience
    change_data.knowledge.setdefault(38,0)
    change_data.knowledge[38] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_MATHEMATICS_EXPERIENCE)
def handle_add_large_mathematics(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量数学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(38,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[38]
    character_data.knowledge[38] += experience
    change_data.knowledge.setdefault(38,0)
    change_data.knowledge[38] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_ASTRONOMY_EXPERIENCE)
def handle_add_small_astronomy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量天文学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(39,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[39]
    character_data.knowledge[39] += experience
    change_data.knowledge.setdefault(39,0)
    change_data.knowledge[39] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_ASTRONOMY_EXPERIENCE)
def handle_add_medium_astronomy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量天文学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(39,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[39]
    character_data.knowledge[39] += experience
    change_data.knowledge.setdefault(39,0)
    change_data.knowledge[39] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_ASTRONOMY_EXPERIENCE)
def handle_add_large_astronomy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量天文学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(39,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[39]
    character_data.knowledge[39] += experience
    change_data.knowledge.setdefault(39,0)
    change_data.knowledge[39] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_PHYSICS_EXPERIENCE)
def handle_add_small_physics(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量物理学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(40,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[40]
    character_data.knowledge[40] += experience
    change_data.knowledge.setdefault(40,0)
    change_data.knowledge[40] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_PHYSICS_EXPERIENCE)
def handle_add_medium_physics(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量物理学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(40,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[40]
    character_data.knowledge[40] += experience
    change_data.knowledge.setdefault(40,0)
    change_data.knowledge[40] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_PHYSICS_EXPERIENCE)
def handle_add_large_physics(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量物理学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(40,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[40]
    character_data.knowledge[40] += experience
    change_data.knowledge.setdefault(40,0)
    change_data.knowledge[40] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_GEOGRAPHY_EXPERIENCE)
def handle_add_small_geography(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量地理学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(41,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[41]
    character_data.knowledge[41] += experience
    change_data.knowledge.setdefault(41,0)
    change_data.knowledge[41] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_GEOGRAPHY_EXPERIENCE)
def handle_add_medium_geography(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量地理学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(41,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[41]
    character_data.knowledge[41] += experience
    change_data.knowledge.setdefault(41,0)
    change_data.knowledge[41] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_GEOGRAPHY_EXPERIENCE)
def handle_add_large_geography(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量地理学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(41,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[41]
    character_data.knowledge[41] += experience
    change_data.knowledge.setdefault(41,0)
    change_data.knowledge[41] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_GEOLOGY_EXPERIENCE)
def handle_add_small_geology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量地质学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(42,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[42]
    character_data.knowledge[42] += experience
    change_data.knowledge.setdefault(42,0)
    change_data.knowledge[42] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_GEOLOGY_EXPERIENCE)
def handle_add_medium_geology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量地质学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(42,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[42]
    character_data.knowledge[42] += experience
    change_data.knowledge.setdefault(42,0)
    change_data.knowledge[42] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_GEOLOGY_EXPERIENCE)
def handle_add_large_geology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量地质学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(42,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[42]
    character_data.knowledge[42] += experience
    change_data.knowledge.setdefault(42,0)
    change_data.knowledge[42] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_ECOLOGY_EXPERIENCE)
def handle_add_small_ecology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量生态学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(43,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[43]
    character_data.knowledge[43] += experience
    change_data.knowledge.setdefault(43,0)
    change_data.knowledge[43] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_ECOLOGY_EXPERIENCE)
def handle_add_medium_ecology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量生态学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(43,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[43]
    character_data.knowledge[43] += experience
    change_data.knowledge.setdefault(43,0)
    change_data.knowledge[43] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_ECOLOGY_EXPERIENCE)
def handle_add_large_ecology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量生态学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(43,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[43]
    character_data.knowledge[43] += experience
    change_data.knowledge.setdefault(43,0)
    change_data.knowledge[43] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_ZOOLOGY_EXPERIENCE)
def handle_add_small_zoology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量动物学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(44,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[44]
    character_data.knowledge[44] += experience
    change_data.knowledge.setdefault(44,0)
    change_data.knowledge[44] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_ZOOLOGY_EXPERIENCE)
def handle_add_medium_zoology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量动物学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(44,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[44]
    character_data.knowledge[44] += experience
    change_data.knowledge.setdefault(44,0)
    change_data.knowledge[44] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_ZOOLOGY_EXPERIENCE)
def handle_add_large_zoology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量动物学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(44,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[44]
    character_data.knowledge[44] += experience
    change_data.knowledge.setdefault(44,0)
    change_data.knowledge[44] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_BOTANY_EXPERIENCE)
def handle_add_small_botany(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量植物学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(45,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[45]
    character_data.knowledge[45] += experience
    change_data.knowledge.setdefault(45,0)
    change_data.knowledge[45] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_BOTANY_EXPERIENCE)
def handle_add_medium_botany(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量植物学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(45,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[45]
    character_data.knowledge[45] += experience
    change_data.knowledge.setdefault(45,0)
    change_data.knowledge[45] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_BOTANY_EXPERIENCE)
def handle_add_large_botany(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量植物学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(45,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[45]
    character_data.knowledge[45] += experience
    change_data.knowledge.setdefault(45,0)
    change_data.knowledge[45] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_ENTOMOLOGY_EXPERIENCE)
def handle_add_small_entomology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量昆虫学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(46,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[46]
    character_data.knowledge[46] += experience
    change_data.knowledge.setdefault(46,0)
    change_data.knowledge[46] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_ENTOMOLOGY_EXPERIENCE)
def handle_add_medium_entomology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量昆虫学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(46,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[46]
    character_data.knowledge[46] += experience
    change_data.knowledge.setdefault(46,0)
    change_data.knowledge[46] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_ENTOMOLOGY_EXPERIENCE)
def handle_add_large_entomology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量昆虫学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(46,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[46]
    character_data.knowledge[46] += experience
    change_data.knowledge.setdefault(46,0)
    change_data.knowledge[46] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_MICROBIOLOGY_EXPERIENCE)
def handle_add_small_microbiology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量微生物学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(47,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[47]
    character_data.knowledge[47] += experience
    change_data.knowledge.setdefault(47,0)
    change_data.knowledge[47] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_MICROBIOLOGY_EXPERIENCE)
def handle_add_medium_microbiology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量微生物学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(47,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[47]
    character_data.knowledge[47] += experience
    change_data.knowledge.setdefault(47,0)
    change_data.knowledge[47] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_MICROBIOLOGY_EXPERIENCE)
def handle_add_large_microbiology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量微生物学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(47,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[47]
    character_data.knowledge[47] += experience
    change_data.knowledge.setdefault(47,0)
    change_data.knowledge[47] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_VIROLOGY_EXPERIENCE)
def handle_add_small_virology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量病毒学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(48,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[48]
    character_data.knowledge[48] += experience
    change_data.knowledge.setdefault(48,0)
    change_data.knowledge[48] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_VIROLOGY_EXPERIENCE)
def handle_add_medium_virology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量病毒学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(48,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[48]
    character_data.knowledge[48] += experience
    change_data.knowledge.setdefault(48,0)
    change_data.knowledge[48] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_VIROLOGY_EXPERIENCE)
def handle_add_large_virology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量病毒学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(48,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[48]
    character_data.knowledge[48] += experience
    change_data.knowledge.setdefault(48,0)
    change_data.knowledge[48] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_BECTERIOLOGY_EXPERIENCE)
def handle_add_small_becteriology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量细菌学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(49,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[49]
    character_data.knowledge[49] += experience
    change_data.knowledge.setdefault(49,0)
    change_data.knowledge[49] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_BECTERIOLOGY_EXPERIENCE)
def handle_add_medium_becteriology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量细菌学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(49,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[49]
    character_data.knowledge[49] += experience
    change_data.knowledge.setdefault(49,0)
    change_data.knowledge[49] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_BECTERIOLOGY_EXPERIENCE)
def handle_add_large_becteriology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量细菌学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(49,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[49]
    character_data.knowledge[49] += experience
    change_data.knowledge.setdefault(49,0)
    change_data.knowledge[49] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_MYCOLOGY_EXPERIENCE)
def handle_add_small_mycology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量真菌学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(50,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[50]
    character_data.knowledge[50] += experience
    change_data.knowledge.setdefault(50,0)
    change_data.knowledge[50] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_MYCOLOGY_EXPERIENCE)
def handle_add_medium_mycology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量真菌学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(50,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[50]
    character_data.knowledge[50] += experience
    change_data.knowledge.setdefault(50,0)
    change_data.knowledge[50] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_MYCOLOGY_EXPERIENCE)
def handle_add_large_mycology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量真菌学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(50,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[50]
    character_data.knowledge[50] += experience
    change_data.knowledge.setdefault(50,0)
    change_data.knowledge[50] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_PHARMACY_EXPERIENCE)
def handle_add_small_pharmacy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量药学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(51,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[51]
    character_data.knowledge[51] += experience
    change_data.knowledge.setdefault(51,0)
    change_data.knowledge[51] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_PHARMACY_EXPERIENCE)
def handle_add_medium_pharmacy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量药学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(51,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[51]
    character_data.knowledge[51] += experience
    change_data.knowledge.setdefault(51,0)
    change_data.knowledge[51] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_PHARMACY_EXPERIENCE)
def handle_add_large_pharmacy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量药学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(51,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[51]
    character_data.knowledge[51] += experience
    change_data.knowledge.setdefault(51,0)
    change_data.knowledge[51] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_METEOROLOGY_EXPERIENCE)
def handle_add_small_meteorology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量气象学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(52,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[52]
    character_data.knowledge[52] += experience
    change_data.knowledge.setdefault(52,0)
    change_data.knowledge[52] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_METEOROLOGY_EXPERIENCE)
def handle_add_medium_meteorology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量气象学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(52,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[52]
    character_data.knowledge[52] += experience
    change_data.knowledge.setdefault(52,0)
    change_data.knowledge[52] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_METEOROLOGY_EXPERIENCE)
def handle_add_large_meteorology(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量气象学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(52,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[52]
    character_data.knowledge[52] += experience
    change_data.knowledge.setdefault(52,0)
    change_data.knowledge[52] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_LAW_SCIENCE_EXPERIENCE)
def handle_add_small_law_science(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量法学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(53,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[53]
    character_data.knowledge[53] += experience
    change_data.knowledge.setdefault(53,0)
    change_data.knowledge[53] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_LAW_SCIENCE_EXPERIENCE)
def handle_add_medium_law_science(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量法学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(53,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[53]
    character_data.knowledge[53] += experience
    change_data.knowledge.setdefault(53,0)
    change_data.knowledge[53] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_LAW_SCIENCE_EXPERIENCE)
def handle_add_large_law_science(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量法学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(53,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[53]
    character_data.knowledge[53] += experience
    change_data.knowledge.setdefault(53,0)
    change_data.knowledge[53] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_LINGUISTICS_EXPERIENCE)
def handle_add_small_linguistics(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量语言学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(54,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[54]
    character_data.knowledge[54] += experience
    change_data.knowledge.setdefault(54,0)
    change_data.knowledge[54] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_LINGUISTICS_EXPERIENCE)
def handle_add_medium_linguistics(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量语言学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(54,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[54]
    character_data.knowledge[54] += experience
    change_data.knowledge.setdefault(54,0)
    change_data.knowledge[54] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_LINGUISTICS_EXPERIENCE)
def handle_add_large_linguistics(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量语言学经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(54,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[54]
    character_data.knowledge[54] += experience
    change_data.knowledge.setdefault(54,0)
    change_data.knowledge[54] += experience
