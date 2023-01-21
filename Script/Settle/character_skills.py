from Script.Design import settle_behavior, constant
from Script.Core import cache_control, game_type


cache: game_type.Cache = cache_control.cache


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_SEXUAL_EXPERIENCE)
def handle_add_small_sexual(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量性爱经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(9,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[9]
    character_data.knowledge[9] += experience
    change_data.knowledge.setdefault(9,0)
    change_data.knowledge[9] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_SEXUAL_EXPERIENCE)
def handle_add_medium_sexual(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量性爱经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(9,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[9]
    character_data.knowledge[9] += experience
    change_data.knowledge.setdefault(9,0)
    change_data.knowledge[9] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_SEXUAL_EXPERIENCE)
def handle_add_large_sexual(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量性爱经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(9,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[9]
    character_data.knowledge[9] += experience
    change_data.knowledge.setdefault(9,0)
    change_data.knowledge[9] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_COMPUTER_EXPERIENCE)
def handle_add_small_computer(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量计算机经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(10,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[10]
    character_data.knowledge[10] += experience
    change_data.knowledge.setdefault(10,0)
    change_data.knowledge[10] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_COMPUTER_EXPERIENCE)
def handle_add_medium_computer(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量计算机经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(10,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[10]
    character_data.knowledge[10] += experience
    change_data.knowledge.setdefault(10,0)
    change_data.knowledge[10] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_COMPUTER_EXPERIENCE)
def handle_add_large_computer(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量计算机经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(10,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[10]
    character_data.knowledge[10] += experience
    change_data.knowledge.setdefault(10,0)
    change_data.knowledge[10] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_PERFORMANCE_EXPERIENCE)
def handle_add_small_performance(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量表演经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(11,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[11]
    character_data.knowledge[11] += experience
    change_data.knowledge.setdefault(11,0)
    change_data.knowledge[11] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_PERFORMANCE_EXPERIENCE)
def handle_add_medium_performance(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量表演经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(11,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[11]
    character_data.knowledge[11] += experience
    change_data.knowledge.setdefault(11,0)
    change_data.knowledge[11] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_PERFORMANCE_EXPERIENCE)
def handle_add_large_performance(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量表演经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(11,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[11]
    character_data.knowledge[11] += experience
    change_data.knowledge.setdefault(11,0)
    change_data.knowledge[11] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_ELOQUENCE_EXPERIENCE)
def handle_add_small_eloquence(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量口才经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(12,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[12]
    character_data.knowledge[12] += experience
    change_data.knowledge.setdefault(12,0)
    change_data.knowledge[12] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_ELOQUENCE_EXPERIENCE)
def handle_add_medium_eloquence(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量口才经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(12,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[12]
    character_data.knowledge[12] += experience
    change_data.knowledge.setdefault(12,0)
    change_data.knowledge[12] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_ELOQUENCE_EXPERIENCE)
def handle_add_large_eloquence(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量口才经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(12,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[12]
    character_data.knowledge[12] += experience
    change_data.knowledge.setdefault(12,0)
    change_data.knowledge[12] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_PAINTING_EXPERIENCE)
def handle_add_small_painting(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量绘画经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(13,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[13]
    character_data.knowledge[13] += experience
    change_data.knowledge.setdefault(13,0)
    change_data.knowledge[13] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_PAINTING_EXPERIENCE)
def handle_add_medium_painting(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量绘画经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(13,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[13]
    character_data.knowledge[13] += experience
    change_data.knowledge.setdefault(13,0)
    change_data.knowledge[13] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_PAINTING_EXPERIENCE)
def handle_add_large_painting(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量绘画经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(13,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[13]
    character_data.knowledge[13] += experience
    change_data.knowledge.setdefault(13,0)
    change_data.knowledge[13] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_SHOOT_EXPERIENCE)
def handle_add_small_shoot(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量拍摄经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(14,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[14]
    character_data.knowledge[14] += experience
    change_data.knowledge.setdefault(14,0)
    change_data.knowledge[14] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_SHOOT_EXPERIENCE)
def handle_add_medium_shoot(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量拍摄经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(14,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[14]
    character_data.knowledge[14] += experience
    change_data.knowledge.setdefault(14,0)
    change_data.knowledge[14] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_SHOOT_EXPERIENCE)
def handle_add_large_shoot(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量拍摄经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(14,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[14]
    character_data.knowledge[14] += experience
    change_data.knowledge.setdefault(14,0)
    change_data.knowledge[14] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_SINGING_EXPERIENCE)
def handle_add_small_singing(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量演唱经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(15,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[15]
    character_data.knowledge[15] += experience
    change_data.knowledge.setdefault(15,0)
    change_data.knowledge[15] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_SINGING_EXPERIENCE)
def handle_add_medium_singing(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量演唱经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(15,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[15]
    character_data.knowledge[15] += experience
    change_data.knowledge.setdefault(15,0)
    change_data.knowledge[15] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_SINGING_EXPERIENCE)
def handle_add_large_singing(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量演唱经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(15,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[15]
    character_data.knowledge[15] += experience
    change_data.knowledge.setdefault(15,0)
    change_data.knowledge[15] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_WRITE_MUSIC_EXPERIENCE)
def handle_add_small_write_music(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量作曲经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(16,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[16]
    character_data.knowledge[16] += experience
    change_data.knowledge.setdefault(16,0)
    change_data.knowledge[16] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_WRITE_MUSIC_EXPERIENCE)
def handle_add_medium_write_music(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量作曲经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(16,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[16]
    character_data.knowledge[16] += experience
    change_data.knowledge.setdefault(16,0)
    change_data.knowledge[16] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_WRITE_MUSIC_EXPERIENCE)
def handle_add_large_write_music(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量作曲经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(16,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[16]
    character_data.knowledge[16] += experience
    change_data.knowledge.setdefault(16,0)
    change_data.knowledge[16] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_COOKING_EXPERIENCE)
def handle_add_small_cooking(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量厨艺经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(17,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[17]
    character_data.knowledge[17] += experience
    change_data.knowledge.setdefault(17,0)
    change_data.knowledge[17] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_COOKING_EXPERIENCE)
def handle_add_medium_cooking(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量厨艺经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(17,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[17]
    character_data.knowledge[17] += experience
    change_data.knowledge.setdefault(17,0)
    change_data.knowledge[17] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_COOKING_EXPERIENCE)
def handle_add_large_cooking(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量厨艺经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(17,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[17]
    character_data.knowledge[17] += experience
    change_data.knowledge.setdefault(17,0)
    change_data.knowledge[17] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_DANCE_EXPERIENCE)
def handle_add_small_dance(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量舞蹈经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(18,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[18]
    character_data.knowledge[18] += experience
    change_data.knowledge.setdefault(18,0)
    change_data.knowledge[18] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_DANCE_EXPERIENCE)
def handle_add_medium_dance(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量舞蹈经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(18,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[18]
    character_data.knowledge[18] += experience
    change_data.knowledge.setdefault(18,0)
    change_data.knowledge[18] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_DANCE_EXPERIENCE)
def handle_add_large_dance(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量舞蹈经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(18,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[18]
    character_data.knowledge[18] += experience
    change_data.knowledge.setdefault(18,0)
    change_data.knowledge[18] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_TAILOR_EXPERIENCE)
def handle_add_small_tailor(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量裁缝经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(19,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[19]
    character_data.knowledge[19] += experience
    change_data.knowledge.setdefault(19,0)
    change_data.knowledge[19] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_TAILOR_EXPERIENCE)
def handle_add_medium_tailor(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量裁缝经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(19,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[19]
    character_data.knowledge[19] += experience
    change_data.knowledge.setdefault(19,0)
    change_data.knowledge[19] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_TAILOR_EXPERIENCE)
def handle_add_large_tailor(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量裁缝经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(19,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[19]
    character_data.knowledge[19] += experience
    change_data.knowledge.setdefault(19,0)
    change_data.knowledge[19] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_TACTICS_EXPERIENCE)
def handle_add_small_tactics(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量战术经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(20,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[20]
    character_data.knowledge[20] += experience
    change_data.knowledge.setdefault(20,0)
    change_data.knowledge[20] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_TACTICS_EXPERIENCE)
def handle_add_medium_tactics(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量战术经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(20,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[20]
    character_data.knowledge[20] += experience
    change_data.knowledge.setdefault(20,0)
    change_data.knowledge[20] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_TACTICS_EXPERIENCE)
def handle_add_large_tactics(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量战术经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(20,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[20]
    character_data.knowledge[20] += experience
    change_data.knowledge.setdefault(20,0)
    change_data.knowledge[20] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_SWIMMING_EXPERIENCE)
def handle_add_small_swimming(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量游泳经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(21,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[21]
    character_data.knowledge[21] += experience
    change_data.knowledge.setdefault(21,0)
    change_data.knowledge[21] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_SWIMMING_EXPERIENCE)
def handle_add_medium_swimming(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量游泳经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(21,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[21]
    character_data.knowledge[21] += experience
    change_data.knowledge.setdefault(21,0)
    change_data.knowledge[21] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_SWIMMING_EXPERIENCE)
def handle_add_large_swimming(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量游泳经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(21,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[21]
    character_data.knowledge[21] += experience
    change_data.knowledge.setdefault(21,0)
    change_data.knowledge[21] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_MANUFACTURE_EXPERIENCE)
def handle_add_small_manufacture(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量制造经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(22,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[22]
    character_data.knowledge[22] += experience
    change_data.knowledge.setdefault(22,0)
    change_data.knowledge[22] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_MANUFACTURE_EXPERIENCE)
def handle_add_medium_manufacture(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量制造经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(22,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[22]
    character_data.knowledge[22] += experience
    change_data.knowledge.setdefault(22,0)
    change_data.knowledge[22] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_MANUFACTURE_EXPERIENCE)
def handle_add_large_manufacture(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量制造经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(22,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[22]
    character_data.knowledge[22] += experience
    change_data.knowledge.setdefault(22,0)
    change_data.knowledge[22] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_FIRST_AID_EXPERIENCE)
def handle_add_small_first_aid(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量急救经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(23,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[23]
    character_data.knowledge[23] += experience
    change_data.knowledge.setdefault(23,0)
    change_data.knowledge[23] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_FIRST_AID_EXPERIENCE)
def handle_add_medium_first_aid(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量急救经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(23,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[23]
    character_data.knowledge[23] += experience
    change_data.knowledge.setdefault(23,0)
    change_data.knowledge[23] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_FIRST_AID_EXPERIENCE)
def handle_add_large_first_aid(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量急救经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(23,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[23]
    character_data.knowledge[23] += experience
    change_data.knowledge.setdefault(23,0)
    change_data.knowledge[23] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_ANATOMY_EXPERIENCE)
def handle_add_small_anatomy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量解剖经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(24,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[24]
    character_data.knowledge[24] += experience
    change_data.knowledge.setdefault(24,0)
    change_data.knowledge[24] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_ANATOMY_EXPERIENCE)
def handle_add_medium_anatomy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量解剖经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(24,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[24]
    character_data.knowledge[24] += experience
    change_data.knowledge.setdefault(24,0)
    change_data.knowledge[24] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_ANATOMY_EXPERIENCE)
def handle_add_large_anatomy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量解剖经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(24,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[24]
    character_data.knowledge[24] += experience
    change_data.knowledge.setdefault(24,0)
    change_data.knowledge[24] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_PLAY_MUSIC_EXPERIENCE)
def handle_add_small_play_music(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量演奏经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(25,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[25]
    character_data.knowledge[25] += experience
    change_data.knowledge.setdefault(25,0)
    change_data.knowledge[25] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_PLAY_MUSIC_EXPERIENCE)
def handle_add_medium_play_music(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量演奏经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(25,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[25]
    character_data.knowledge[25] += experience
    change_data.knowledge.setdefault(25,0)
    change_data.knowledge[25] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_PLAY_MUSIC_EXPERIENCE)
def handle_add_large_play_music(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量演奏经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(25,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[25]
    character_data.knowledge[25] += experience
    change_data.knowledge.setdefault(25,0)
    change_data.knowledge[25] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_PROGRAMMING_EXPERIENCE)
def handle_add_small_programming(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量编程经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(26,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[26]
    character_data.knowledge[26] += experience
    change_data.knowledge.setdefault(26,0)
    change_data.knowledge[26] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_PROGRAMMING_EXPERIENCE)
def handle_add_medium_programming(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量编程经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(26,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[26]
    character_data.knowledge[26] += experience
    change_data.knowledge.setdefault(26,0)
    change_data.knowledge[26] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_PROGRAMMING_EXPERIENCE)
def handle_add_large_programming(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量编程经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(26,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[26]
    character_data.knowledge[26] += experience
    change_data.knowledge.setdefault(26,0)
    change_data.knowledge[26] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_HACKER_EXPERIENCE)
def handle_add_small_hacker(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量黑客经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(27,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[27]
    character_data.knowledge[27] += experience
    change_data.knowledge.setdefault(27,0)
    change_data.knowledge[27] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_HACKER_EXPERIENCE)
def handle_add_medium_hacker(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量黑客经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(27,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[27]
    character_data.knowledge[27] += experience
    change_data.knowledge.setdefault(27,0)
    change_data.knowledge[27] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_HACKER_EXPERIENCE)
def handle_add_large_hacker(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量黑客经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(27,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[27]
    character_data.knowledge[27] += experience
    change_data.knowledge.setdefault(27,0)
    change_data.knowledge[27] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_WRITE_EXPERIENCE)
def handle_add_small_write(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量写作经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(28,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[28]
    character_data.knowledge[28] += experience
    change_data.knowledge.setdefault(28,0)
    change_data.knowledge[28] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_WRITE_EXPERIENCE)
def handle_add_medium_write(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量写作经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(28,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[28]
    character_data.knowledge[28] += experience
    change_data.knowledge.setdefault(28,0)
    change_data.knowledge[28] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_WRITE_EXPERIENCE)
def handle_add_large_write(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量写作经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(28,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[28]
    character_data.knowledge[28] += experience
    change_data.knowledge.setdefault(28,0)
    change_data.knowledge[28] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_TRANSACTION_EXPERIENCE)
def handle_add_small_transaction(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量交易经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(29,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[29]
    character_data.knowledge[29] += experience
    change_data.knowledge.setdefault(29,0)
    change_data.knowledge[29] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_TRANSACTION_EXPERIENCE)
def handle_add_medium_transaction(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量交易经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(29,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[29]
    character_data.knowledge[29] += experience
    change_data.knowledge.setdefault(29,0)
    change_data.knowledge[29] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_TRANSACTION_EXPERIENCE)
def handle_add_large_transaction(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量交易经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(29,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[29]
    character_data.knowledge[29] += experience
    change_data.knowledge.setdefault(29,0)
    change_data.knowledge[29] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_CEREMONY_EXPERIENCE)
def handle_add_small_ceremony(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量礼仪经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(30,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[30]
    character_data.knowledge[30] += experience
    change_data.knowledge.setdefault(30,0)
    change_data.knowledge[30] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_CEREMONY_EXPERIENCE)
def handle_add_medium_ceremony(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量礼仪经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(30,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[30]
    character_data.knowledge[30] += experience
    change_data.knowledge.setdefault(30,0)
    change_data.knowledge[30] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_CEREMONY_EXPERIENCE)
def handle_add_large_ceremony(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量礼仪经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(30,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[30]
    character_data.knowledge[30] += experience
    change_data.knowledge.setdefault(30,0)
    change_data.knowledge[30] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_MOTION_EXPERIENCE)
def handle_add_small_motion(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量运动经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(31,0)
    experience = 0.01 * add_time * character_data.knowledge_interest[31]
    character_data.knowledge[31] += experience
    change_data.knowledge.setdefault(31,0)
    change_data.knowledge[31] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_MOTION_EXPERIENCE)
def handle_add_medium_motion(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量运动经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(31,0)
    experience = 0.05 * add_time * character_data.knowledge_interest[31]
    character_data.knowledge[31] += experience
    change_data.knowledge.setdefault(31,0)
    change_data.knowledge[31] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_MOTION_EXPERIENCE)
def handle_add_large_motion(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量运动经验
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.knowledge.setdefault(31,0)
    experience = 0.1 * add_time * character_data.knowledge_interest[31]
    character_data.knowledge[31] += experience
    change_data.knowledge.setdefault(31,0)
    change_data.knowledge[31] += experience
