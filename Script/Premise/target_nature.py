from Script.Design import handle_premise
from Script.Core import constant, game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.TARGET_IS_LIVELY)
def handle_target_is_lively(character_id: int) -> int:
    """
    校验交互对象是否是一个活跃的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[0] >= 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_LOW_KEY)
def handle_target_is_low_key(character_id: int) -> int:
    """
    校验交互对象是否是一个低调的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[0] < 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_GREGARIOUS)
def handle_target_is_gregarious(character_id: int) -> int:
    """
    校验交互对象是否是一个合群的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[1] >= 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_SOLITARY)
def handle_target_is_solitary(character_id: int) -> int:
    """
    校验交互对象是否是一个孤僻的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[1] < 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_OPTIMISTIC)
def handle_target_is_optimisitic(character_id: int) -> int:
    """
    校验交互对象是否是一个乐观的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[2] >= 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_PESSIMISM)
def handle_target_is_pessimism(character_id: int) -> int:
    """
    校验交互对象是否是一个悲观的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[2] < 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_KEEP_PROMISES)
def handle_target_is_keep_promises(character_id: int) -> int:
    """
    校验交互对象是否是一个守信的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[3] >= 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_DECEITFUL)
def handle_target_is_deceitful(character_id: int) -> int:
    """
    校验交互对象是否是一个狡诈的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[3] < 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_SELFLESS)
def handle_target_is_selfless(character_id: int) -> int:
    """
    校验交互对象是否是一个无私的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[4] >= 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_SELFISH)
def handle_target_is_selfish(character_id: int) -> int:
    """
    校验交互对象是否是一个自私的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[4] < 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_HEAVY_FEELING)
def handle_target_is_heavy_feeling(character_id: int) -> int:
    """
    校验交互对象是否是一个重情的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[5] >= 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_UNGRATEFUL)
def handle_target_is_ungrateful(character_id: int) -> int:
    """
    校验交互对象是否是一个薄情的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[5] < 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_RIGOROUS)
def handle_target_is_rigorous(character_id: int) -> int:
    """
    校验交互对象是否是一个严谨的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[6] >= 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_RELAX)
def handle_target_is_relax(character_id: int) -> int:
    """
    校验交互对象是否是一个松散的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[6] < 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_AUTONOMY)
def handle_target_is_autonomy(character_id: int) -> int:
    """
    校验交互对象是否是一个自律的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[7] >= 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_INDULGE)
def handle_target_is_indulge(character_id: int) -> int:
    """
    校验交互对象是否是一个放纵的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[7] < 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_STEADY)
def handle_target_is_steady(character_id: int) -> int:
    """
    校验交互对象是否是一个沉稳的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[8] >= 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_CHILDISH)
def handle_target_is_childish(character_id: int) -> int:
    """
    校验交互对象是否是一个稚拙的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[8] < 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_RESOLUTION)
def handle_target_is_resolution(character_id: int) -> int:
    """
    校验交互对象是否是一个决断的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[9] >= 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_HESITATE)
def handle_target_is_hesitate(character_id: int) -> int:
    """
    校验交互对象是否是一个犹豫的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[9] < 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_TENACITY)
def handle_target_is_tenacity(character_id: int) -> int:
    """
    校验交互对象是否是一个坚韧的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[10] >= 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_FRAGILE)
def handle_target_is_fragile(character_id: int) -> int:
    """
    校验交互对象是否是一个脆弱的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[10] < 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_ASTUTE)
def handle_target_is_astute(character_id: int) -> int:
    """
    校验交互对象是否是一个机敏的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[11] >= 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_SLOW)
def handle_target_is_slow(character_id: int) -> int:
    """
    校验交互对象是否是一个迟钝的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[11] < 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_TOLERANCE)
def handle_target_is_tolerance(character_id: int) -> int:
    """
    校验交互对象是否是一个耐性的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[12] >= 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_IMPETUOUS)
def handle_target_is_impetuous(character_id: int) -> int:
    """
    校验交互对象是否是一个浮躁的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[12] < 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_STARAIGHTFORWARD)
def handle_target_is_staraightforward(character_id: int) -> int:
    """
    校验交互对象是否是一个爽直的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[13] >= 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_INSIDIOUS)
def handle_target_is_insidious(character_id: int) -> int:
    """
    校验交互对象是否是一个阴险的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[13] < 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_TOLERANT)
def handle_target_is_tolerance(character_id: int) -> int:
    """
    校验交互对象是否是一个宽和的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[14] >= 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_NARROW)
def handle_target_is_narrow(character_id: int) -> int:
    """
    校验交互对象是否是一个阴险的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[14] < 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_ENTHUSIASM)
def handle_target_is_enthusiasm(character_id: int) -> int:
    """
    校验交互对象是否是一个热情的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[15] >= 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_APATHY)
def handle_target_is_apathy(character_id: int) -> int:
    """
    校验交互对象是否是一个冷漠的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[15] < 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_SELF_CONFIDENCE)
def handle_target_is_self_confidence(character_id: int) -> int:
    """
    校验交互对象是否是一个自信的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[16] >= 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_INFERIORITY)
def handle_target_is_inferiority(character_id: int) -> int:
    """
    校验交互对象是否是一个自卑的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[16] < 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_INFFERENT)
def handle_target_is_infferent(character_id: int) -> int:
    """
    校验交互对象是否是一个淡泊的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[17] >= 50


@handle_premise.add_premise(constant.Premise.TARGET_IS_KEEN)
def handle_target_is_keen(character_id: int) -> int:
    """
    校验交互对象是否是一个热衷的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return target_data.nature[17] < 50
