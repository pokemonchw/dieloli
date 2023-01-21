from types import FunctionType
from Script.Design import settle_behavior, attr_calculation, constant
from Script.Core import cache_control, game_type, get_text
from Script.Config import normal_config


_: FunctionType = get_text._
""" 翻译api """
window_width: int = normal_config.config_normal.text_width
""" 窗体宽度 """
cache: game_type.Cache = cache_control.cache


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_MOUTH_HAPPY)
def handle_add_small_mouth_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量嘴部快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(0,0)
    character_happy = add_time
    character_data.sex_experience.setdefault(0,0)
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[0]) + character_data.status[0] / 100
    character_data.status[0] += character_happy
    change_data.status.setdefault(0,0)
    change_data.status[0] += character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_MOUTH_HAPPY)
def handle_add_medium_mouth_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量嘴部快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(0,0)
    character_data.sex_experience.setdefault(0,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[0]) + character_data.status[0] / 50
    character_data.status[0] += character_happy
    change_data.status.setdefault(0,0)
    change_data.status[0] += character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_MOUTH_HAPPY)
def handle_add_large_mouth_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量嘴部快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(0,0)
    character_data.sex_experience.setdefault(0,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[0]) + character_data.status[0] / 10
    character_data.status[0] += character_happy
    change_data.status.setdefault(0,0)
    change_data.status[0] += character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_SMALL_MOUTH_HAPPY)
def handle_sub_small_mouth_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少少量嘴部快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(0,0)
    character_data.sex_experience.setdefault(0,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[0]) + character_data.status[0] / 100
    character_data.status[0] -= character_happy
    character_data.status[0] = max(character_data.status[0],0)
    change_data.status.setdefault(0,0)
    change_data.status[0] -= character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_MEDIUM_MOUTH_HAPPY)
def handle_sub_medium_mouth_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少中量嘴部快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(0,0)
    character_data.sex_experience.setdefault(0,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[0]) + character_data.status[0] / 50
    character_data.status[0] -= character_happy
    character_data.status[0] = max(character_data.status[0],0)
    change_data.status.setdefault(0,0)
    change_data.status[0] -= character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_LARGE_MOUTH_HAPPY)
def handle_sub_large_mouth_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少大量嘴部快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(0,0)
    character_data.sex_experience.setdefault(0,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[0]) + character_data.status[0] / 10
    character_data.status[0] -= character_happy
    character_data.status[0] = max(character_data.status[0],0)
    change_data.status.setdefault(0,0)
    change_data.status[0] -= character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_CHEST_HAPPY)
def handle_add_small_chest_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量胸部快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(1,0)
    character_data.sex_experience.setdefault(1,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[1]) + character_data.status[1] / 100
    character_data.status[1] += character_happy
    change_data.status.setdefault(1,0)
    change_data.status[1] += character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_CHEST_HAPPY)
def handle_add_medium_chest_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量胸部快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(1,0)
    character_data.sex_experience.setdefault(1,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[1]) + character_data.status[1] / 50
    character_data.status[1] += character_happy
    change_data.status.setdefault(1,0)
    change_data.status[1] += character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_CHEST_HAPPY)
def handle_add_large_chest_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量胸部快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(1,0)
    character_data.sex_experience.setdefault(1,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[1]) + character_data.status[1] / 10
    character_data.status[1] += character_happy
    change_data.status.setdefault(1,0)
    change_data.status[1] += character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_SMALL_CHEST_HAPPY)
def handle_sub_small_chest_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少少量胸部快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(1,0)
    character_data.sex_experience.setdefault(1,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[1]) + character_data.status[1] / 100
    character_data.status[1] -= character_happy
    character_data.status[1] = max(character_data.status[1],0)
    change_data.status.setdefault(1,0)
    change_data.status[1] -= character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_MEDIUM_CHEST_HAPPY)
def handle_sub_medium_chest_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少中量胸部快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(1,0)
    character_data.sex_experience.setdefault(1,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[1]) + character_data.status[1] / 50
    character_data.status[1] -= character_happy
    character_data.status[1] = max(character_data.status[1],0)
    change_data.status.setdefault(1,0)
    change_data.status[1] -= character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_LARGE_CHEST_HAPPY)
def handle_sub_large_chest_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少大量胸部快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(1,0)
    character_data.sex_experience.setdefault(1,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[1]) + character_data.status[1] / 10
    character_data.status[1] -= character_happy
    character_data.status[1] = max(character_data.status[1],0)
    change_data.status.setdefault(1,0)
    change_data.status[1] -= character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_VAGINA_HAPPY)
def handle_add_small_vagina_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量阴道快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(2,0)
    character_data.sex_experience.setdefault(4,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[4]) + character_data.status[2] / 100
    character_data.status[2] += character_happy
    change_data.status.setdefault(2,0)
    change_data.status[2] += character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_VAGINA_HAPPY)
def handle_add_medium_vagina_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量阴道快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(2,0)
    character_data.sex_experience.setdefault(4,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[4]) + character_data.status[2] / 50
    character_data.status[2] += character_happy
    change_data.status.setdefault(2,0)
    change_data.status[2] += character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_VAGINA_HAPPY)
def handle_add_large_vagina_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量阴道快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(2,0)
    character_data.sex_experience.setdefault(4,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[4]) + character_data.status[2] / 10
    character_data.status[2] += character_happy
    change_data.status.setdefault(2,0)
    change_data.status[2] += character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_SMALL_VAGINA_HAPPY)
def handle_sub_small_vagina_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少少量阴道快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(2,0)
    character_data.sex_experience.setdefault(4,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[4]) + character_data.status[2] / 100
    character_data.status[2] -= character_happy
    character_data.status[2] = max(character_data.status[2],0)
    change_data.status.setdefault(2,0)
    change_data.status[2] -= character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_MEDIUM_VAGINA_HAPPY)
def handle_sub_medium_vagina_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少中量阴道快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(2,0)
    character_data.sex_experience.setdefault(4,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[4]) + character_data.status[2] / 50
    character_data.status[2] -= character_happy
    character_data.status[2] = max(character_data.status[2],0)
    change_data.status.setdefault(2,0)
    change_data.status[2] -= character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_LARGE_VAGINA_HAPPY)
def handle_sub_large_vagina_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少大量阴道快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(2,0)
    character_data.sex_experience.setdefault(4,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[4]) + character_data.status[2] / 10
    character_data.status[2] -= character_happy
    character_data.status[2] = max(character_data.status[2],0)
    change_data.status.setdefault(2,0)
    change_data.status[2] -= character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_CLITORIS_HAPPY)
def handle_add_small_clitoris_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量阴蒂快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(3,0)
    character_data.sex_experience.setdefault(2,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[2]) + character_data.status[3] / 100
    character_data.status[3] += character_happy
    change_data.status.setdefault(3,0)
    change_data.status[3] += character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_CLITORIS_HAPPY)
def handle_add_medium_clitoris_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量阴蒂快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(3,0)
    character_data.sex_experience.setdefault(2,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[2]) + character_data.status[3] / 50
    character_data.status[3] += character_happy
    change_data.status.setdefault(3,0)
    change_data.status[3] += character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_CLITORIS_HAPPY)
def handle_add_large_clitoris_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量阴蒂快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(3,0)
    character_data.sex_experience.setdefault(2,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[2]) + character_data.status[3] / 10
    character_data.status[3] += character_happy
    change_data.status.setdefault(3,0)
    change_data.status[3] += character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_SMALL_CLITORIS_HAPPY)
def handle_sub_small_clitoris_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少少量阴蒂快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(3,0)
    character_data.sex_experience.setdefault(2,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[2]) + character_data.status[3] / 100
    character_data.status[3] -= character_happy
    character_data.status[3] = max(character_data.status[3],0)
    change_data.status.setdefault(3,0)
    change_data.status[3] -= character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_MEDIUM_CLITORIS_HAPPY)
def handle_sub_medium_clitoris_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少中量阴蒂快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(3,0)
    character_data.sex_experience.setdefault(2,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[2]) + character_data.status[3] / 50
    character_data.status[3] -= character_happy
    character_data.status[3] = max(character_data.status[3],0)
    change_data.status.setdefault(3,0)
    change_data.status[3] -= character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_LARGE_CLITORIS_HAPPY)
def handle_sub_large_clitoris_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少大量阴蒂快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(3,0)
    character_data.sex_experience.setdefault(2,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[2]) + character_data.status[3] / 10
    character_data.status[3] -= character_happy
    character_data.status[3] = max(character_data.status[3],0)
    change_data.status.setdefault(3,0)
    change_data.status[3] -= character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_ANUS_HAPPY)
def handle_add_small_anus_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量肛门快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(4,0)
    character_data.sex_experience.setdefault(5,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[5]) + character_data.status[4] / 100
    character_data.status[4] += character_happy
    change_data.status.setdefault(4,0)
    change_data.status[4] += character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_ANUS_HAPPY)
def handle_add_medium_anus_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量肛门快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(4,0)
    character_data.sex_experience.setdefault(5,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[5]) + character_data.status[4] / 50
    character_data.status[4] += character_happy
    change_data.status.setdefault(4,0)
    change_data.status[4] += character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_ANUS_HAPPY)
def handle_add_large_anus_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量肛门快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(4,0)
    character_data.sex_experience.setdefault(5,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[5]) + character_data.status[4] / 10
    character_data.status[4] += character_happy
    change_data.status.setdefault(4,0)
    change_data.status[4] += character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_SMALL_ANUS_HAPPY)
def handle_sub_small_anus_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少少量肛门快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(4,0)
    character_data.sex_experience.setdefault(5,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[5]) + character_data.status[4] / 100
    character_data.status[4] -= character_happy
    character_data.status[4] = max(character_data.status[4],0)
    change_data.status.setdefault(4,0)
    change_data.status[4] -= character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_MEDIUM_ANUS_HAPPY)
def handle_sub_medium_anus_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少中量肛门快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(4,0)
    character_data.sex_experience.setdefault(5,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[5]) + character_data.status[4] / 50
    character_data.status[4] -= character_happy
    character_data.status[4] = max(character_data.status[4],0)
    change_data.status.setdefault(4,0)
    change_data.status[4] -= character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_LARGE_ANUS_HAPPY)
def handle_sub_large_anus_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少大量肛门快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(4,0)
    character_data.sex_experience.setdefault(5,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[5]) + character_data.status[4] / 10
    character_data.status[4] -= character_happy
    character_data.status[4] = max(character_data.status[4],0)
    change_data.status.setdefault(4,0)
    change_data.status[4] -= character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_PENIS_HAPPY)
def handle_add_small_penis_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加少量阴茎快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(5,0)
    character_data.sex_experience.setdefault(3,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[3]) + character_data.status[5] / 100
    character_data.status[5] += character_happy
    change_data.status.setdefault(5,0)
    change_data.status[5] += character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_PENIS_HAPPY)
def handle_add_medium_penis_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加中量阴茎快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(5,0)
    character_data.sex_experience.setdefault(3,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[3]) + character_data.status[5] / 50
    character_data.status[5] += character_happy
    change_data.status.setdefault(5,0)
    change_data.status[5] += character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_PENIS_HAPPY)
def handle_add_large_penis_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色增加大量阴茎快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(5,0)
    character_data.sex_experience.setdefault(3,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[3]) + character_data.status[5] / 10
    character_data.status[5] += character_happy
    change_data.status.setdefault(5,0)
    change_data.status[5] += character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_SMALL_PENIS_HAPPY)
def handle_sub_small_penis_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少少量阴茎快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(5,0)
    character_data.sex_experience.setdefault(3,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[3]) + character_data.status[5] / 100
    character_data.status[5] -= character_happy
    character_data.status[5] = max(character_data.status[5],0)
    change_data.status.setdefault(5,0)
    change_data.status[5] -= character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_MEDIUM_PENIS_HAPPY)
def handle_sub_medium_penis_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少中量阴茎快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(5,0)
    character_data.sex_experience.setdefault(3,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[3]) + character_data.status[5] / 50
    character_data.status[5] -= character_happy
    character_data.status[5] = max(character_data.status[5],0)
    change_data.status.setdefault(5,0)
    change_data.status[5] -= character_happy


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_LARGE_PENIS_HAPPY)
def handle_sub_large_penis_happy(character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int):
    """
    角色减少大量阴茎快感快感
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    character_data.status.setdefault(5,0)
    character_data.sex_experience.setdefault(3,0)
    character_happy = add_time
    character_happy *= 1 + attr_calculation.get_experience_level_weight(character_data.sex_experience[3]) + character_data.status[5] / 10
    character_data.status[5] -= character_happy
    character_data.status[5] = max(character_data.status[5],0)
    change_data.status.setdefault(5,0)
    change_data.status[5] -= character_happy
