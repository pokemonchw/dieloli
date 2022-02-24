from Script.Design import settle_behavior
from Script.Core import cache_control, constant, game_type


cache: game_type.Cache = cache_control.cache


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_ETHIC_EXPERIENCE)
def handle_target_add_small_ethic(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量伦理经验
    Keyword arguments:
    character_id -- 角色id
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
    experience = 0.01 * add_time * target_data.knowledge_interest[0]
    target_data.knowledge.setdefault(0,0)
    target_data.knowledge[0] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_ETHIC_EXPERIENCE)
def handle_target_add_medium_ethic(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量伦理经验
    Keyword arguments:
    character_id -- 角色id
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
    experience = 0.05 * add_time * target_data.knowledge_interest[0]
    target_data.knowledge.setdefault(0,0)
    target_data.knowledge[0] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_ETHIC_EXPERIENCE)
def handle_target_add_large_ethic(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量伦理经验
    Keyword arguments:
    character_id -- 角色id
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
    experience = 0.1 * add_time * target_data.knowledge_interest[0]
    target_data.knowledge.setdefault(0,0)
    target_data.knowledge[0] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_MORALITY_EXPERIENCE)
def handle_target_add_small_morality(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量道德经验
    Keyword arguments:
    character_id -- 角色id
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
    experience = 0.01 * add_time * target_data.knowledge_interest[1]
    target_data.knowledge.setdefault(1,0)
    target_data.knowledge[1] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_MORALITY_EXPERIENCE)
def handle_target_add_medium_morality(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量道德经验
    Keyword arguments:
    character_id -- 角色id
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
    experience = 0.05 * add_time * target_data.knowledge_interest[1]
    target_data.knowledge.setdefault(1,0)
    target_data.knowledge[1] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_MORALITY_EXPERIENCE)
def handle_target_add_large_morality(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量道德经验
    Keyword arguments:
    character_id -- 角色id
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
    experience = 0.1 * add_time * target_data.knowledge_interest[1]
    target_data.knowledge.setdefault(1,0)
    target_data.knowledge[1] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_LITERATURE_EXPERIENCE)
def handle_target_add_small_literature(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量文学经验
    Keyword arguments:
    character_id -- 角色id
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
    experience = 0.01 * add_time * target_data.knowledge_interest[2]
    target_data.knowledge.setdefault(2,0)
    target_data.knowledge[2] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_LITERATURE_EXPERIENCE)
def handle_target_add_medium_literature(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量文学经验
    Keyword arguments:
    character_id -- 角色id
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
    experience = 0.05 * add_time * target_data.knowledge_interest[2]
    target_data.knowledge.setdefault(2,0)
    target_data.knowledge[2] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_LITERATURE_EXPERIENCE)
def handle_target_add_large_literature(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量文学经验
    Keyword arguments:
    character_id -- 角色id
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
    experience = 0.1 * add_time * target_data.knowledge_interest[2]
    target_data.knowledge.setdefault(2,0)
    target_data.knowledge[2] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_POETRY_EXPERIENCE)
def handle_target_add_small_poetry(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量诗歌经验
    Keyword arguments:
    character_id -- 角色id
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
    experience = 0.01 * add_time * target_data.knowledge_interest[3]
    target_data.knowledge.setdefault(3,0)
    target_data.knowledge[3] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_POETRY_EXPERIENCE)
def handle_target_add_medium_poetry(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量诗歌经验
    Keyword arguments:
    character_id -- 角色id
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
    experience = 0.05 * add_time * target_data.knowledge_interest[3]
    target_data.knowledge.setdefault(3,0)
    target_data.knowledge[3] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_POETRY_EXPERIENCE)
def handle_target_add_large_poetry(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量诗歌经验
    Keyword arguments:
    character_id -- 角色id
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
    experience = 0.1 * add_time * target_data.knowledge_interest[3]
    target_data.knowledge.setdefault(3,0)
    target_data.knowledge[3] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_HISTORY_EXPERIENCE)
def handle_target_add_small_history(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量历史经验
    Keyword arguments:
    character_id -- 角色id
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
    experience = 0.01 * add_time * target_data.knowledge_interest[4]
    target_data.knowledge.setdefault(4,0)
    target_data.knowledge[4] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_HISTORY_EXPERIENCE)
def handle_target_add_medium_history(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量历史经验
    Keyword arguments:
    character_id -- 角色id
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
    experience = 0.05 * add_time * target_data.knowledge_interest[4]
    target_data.knowledge.setdefault(4,0)
    target_data.knowledge[4] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_HISTORY_EXPERIENCE)
def handle_target_add_large_history(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量历史经验
    Keyword arguments:
    character_id -- 角色id
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
    experience = 0.1 * add_time * target_data.knowledge_interest[4]
    target_data.knowledge.setdefault(4,0)
    target_data.knowledge[4] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_ART_EXPERIENCE)
def handle_target_add_small_art(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量艺术经验
    Keyword arguments:
    character_id -- 角色id
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
    experience = 0.01 * add_time * target_data.knowledge_interest[5]
    target_data.knowledge.setdefault(5,0)
    target_data.knowledge[5] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_ART_EXPERIENCE)
def handle_target_add_medium_art(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量艺术经验
    Keyword arguments:
    character_id -- 角色id
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
    experience = 0.05 * add_time * target_data.knowledge_interest[5]
    target_data.knowledge.setdefault(5,0)
    target_data.knowledge[5] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_ART_EXPERIENCE)
def handle_target_add_large_art(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量艺术经验
    Keyword arguments:
    character_id -- 角色id
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
    experience = 0.1 * add_time * target_data.knowledge_interest[5]
    target_data.knowledge.setdefault(5,0)
    target_data.knowledge[5] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_MUSIC_EXPERIENCE)
def handle_target_add_small_music(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量乐理经验
    Keyword arguments:
    character_id -- 角色id
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
    experience = 0.01 * add_time * target_data.knowledge_interest[6]
    target_data.knowledge.setdefault(6,0)
    target_data.knowledge[6] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_MUSIC_EXPERIENCE)
def handle_target_add_medium_music(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量乐理经验
    Keyword arguments:
    character_id -- 角色id
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
    experience = 0.05 * add_time * target_data.knowledge_interest[6]
    target_data.knowledge.setdefault(6,0)
    target_data.knowledge[6] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_MUSIC_EXPERIENCE)
def handle_target_add_large_music(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量乐理经验
    Keyword arguments:
    character_id -- 角色id
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
    experience = 0.1 * add_time * target_data.knowledge_interest[6]
    target_data.knowledge.setdefault(6,0)
    target_data.knowledge[6] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_RELIGION_EXPERIENCE)
def handle_target_add_small_religion(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量宗教经验
    Keyword arguments:
    character_id -- 角色id
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
    experience = 0.01 * add_time * target_data.knowledge_interest[7]
    target_data.knowledge.setdefault(7,0)
    target_data.knowledge[7] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_RELIGION_EXPERIENCE)
def handle_target_add_medium_religion(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量宗教经验
    Keyword arguments:
    character_id -- 角色id
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
    experience = 0.05 * add_time * target_data.knowledge_interest[7]
    target_data.knowledge.setdefault(7,0)
    target_data.knowledge[7] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_RELIGION_EXPERIENCE)
def handle_target_add_large_religion(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量宗教经验
    Keyword arguments:
    character_id -- 角色id
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
    experience = 0.1 * add_time * target_data.knowledge_interest[7]
    target_data.knowledge.setdefault(7,0)
    target_data.knowledge[7] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_SMALL_FAITH_EXPERIENCE)
def handle_target_add_small_faith(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加少量信仰经验
    Keyword arguments:
    character_id -- 角色id
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
    experience = 0.01 * add_time * target_data.knowledge_interest[8]
    target_data.knowledge.setdefault(8,0)
    target_data.knowledge[8] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_MEDIUM_FAITH_EXPERIENCE)
def handle_target_add_medium_faith(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加中量信仰经验
    Keyword arguments:
    character_id -- 角色id
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
    experience = 0.05 * add_time * target_data.knowledge_interest[8]
    target_data.knowledge.setdefault(8,0)
    target_data.knowledge[8] += experience


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_ADD_LARGE_FAITH_EXPERIENCE)
def handle_target_add_large_faith(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    交互对象增加大量信仰经验
    Keyword arguments:
    character_id -- 角色id
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
    experience = 0.1 * add_time * target_data.knowledge_interest[8]
    target_data.knowledge.setdefault(8,0)
    target_data.knowledge[8] += experience
