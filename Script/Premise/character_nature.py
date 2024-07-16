from Script.Design import handle_premise, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.IS_LIVELY)
def handle_is_lively(character_id: int) -> int:
    """
    校验角色是否是一个活跃的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[0] >= 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_LOW_KEY)
def handle_is_low_key(character_id: int) -> int:
    """
    校验角色是否是一个低调的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[0] < 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_GREGARIOUS)
def handle_is_gregarious(character_id: int) -> int:
    """
    校验角色是否是一个合群的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
def handle_target_is_heavy_feeling(character_id: int) -> int:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[1] >= 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_SOLITARY)
def handle_is_solitary(character_id: int) -> int:
    """
    校验角色是否是一个孤僻的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[1] < 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_OPTIMISTIC)
def handle_is_optimistic(character_id: int) -> int:
    """
    校验角色是否是一个乐观的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[2] >= 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_PESSIMISM)
def handle_is_pessimism(character_id: int) -> int:
    """
    校验角色是否是一个悲观的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[2] < 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_KEEP_PROMISES)
def handle_is_keep_promises(character_id: int) -> int:
    """
    校验角色是否是一个守信的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[3] >= 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_DECEITFUL)
def handle_is_deceitful(character_id: int) -> int:
    """
    校验角色是否是一个狡诈的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[3] < 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_SELFLESS)
def handle_is_selfless(character_id: int) -> int:
    """
    校验角色是否是一个无私的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[4] >= 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_SELFISH)
def handle_is_selfish(character_id: int) -> int:
    """
    校验角色是否是一个自私的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[4] < 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_HEAVY_FEELING)
def handle_is_heavy_feeling(character_id: int) -> int:
    """
    校验是否是一个重情的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[5] >= 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_UNGRATEFUL)
def handle_is_ungrateful(character_id: int) -> int:
    """
    校验角色是否是一个薄情的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[5] < 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_RIGOROUS)
def handle_is_rigorous(character_id: int) -> int:
    """
    校验角色是否是一个严谨的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[6] >= 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_RELAX)
def handle_is_relax(character_id: int) -> int:
    """
    校验角色是否是一个松散的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[6] < 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_AUTONOMY)
def handle_is_autonomy(character_id: int) -> int:
    """
    校验角色是否是一个自律的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[7] >= 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_INDULGE)
def handle_is_indulge(character_id: int) -> int:
    """
    校验角色是否是一个放纵的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[7] < 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_STEADY)
def handle_is_steady(character_id: int) -> int:
    """
    校验角色是否是一个沉稳的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[8] >= 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_CHILDISH)
def handle_is_childish(character_id: int) -> int:
    """
    校验角色是否是一个稚拙的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[8] < 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_RESOLUTION)
def handle_is_resolution(character_id: int) -> int:
    """
    校验角色是否是一个决断的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[9] >= 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_HESITATE)
def handle_is_hesitate(character_id: int) -> int:
    """
    校验角色是否是一个犹豫的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[9] < 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_TENACITY)
def handle_is_tenacity(character_id: int) -> int:
    """
    校验角色是否是一个坚韧的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[10] >= 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_FRAGILE)
def handle_is_fragile(character_id: int) -> int:
    """
    校验角色是否是一个脆弱的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[10] < 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_ASTUTE)
def handle_is_astute(character_id: int) -> int:
    """
    校验是否是一个机敏的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[11] >= 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_SLOW)
def handle_is_slow(character_id: int) -> int:
    """
    校验角色是否是一个迟钝的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[11] < 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_TOLERANCE)
def handle_is_tolerance(character_id: int) -> int:
    """
    校验角色是否是一个耐性的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[12] >= 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_IMPETUOUS)
def handle_is_impetuous(character_id: int) -> int:
    """
    校验角色是否是一个浮躁的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[12] < 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_STARAIGHTFORWARD)
def handle_is_staraightforward(character_id: int) -> int:
    """
    校验角色是否是一个爽直的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[13] >= 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_INSIDIOUS)
def handle_is_insidious(character_id: int) -> int:
    """
    校验角色是否是一个阴险的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[13] < 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_TOLERANT)
def handle_is_tolerant(character_id: int) -> int:
    """
    校验角色是否是一个宽和的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[14] >= 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_NARROW)
def handle_is_narrow(character_id: int) -> int:
    """
    校验角色是否是一个狭隘的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[14] < 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_ENTHUSIASM)
def handle_is_enthusiasm(character_id: int) -> int:
    """
    校验角色是否是一个热情的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[15] >= 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_APATHY)
def handle_is_apathy(character_id: int) -> int:
    """
    校验角色是否是一个冷漠的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[15] < 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_SELF_CONFIDENCE)
def handle_is_self_confidence(character_id: int) -> int:
    """
    校验角色是否是一个自信的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[16] >= 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_INFERIORITY)
def handle_is_inferiority(character_id: int) -> int:
    """
    校验角色是否是一个自卑的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[16] < 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_INFFERENT)
def handle_is_infferent(character_id: int) -> int:
    """
    校验角色是否是一个淡泊的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[17] >= 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_KEEN)
def handle_is_keen(character_id: int) -> int:
    """
    校验角色是否是一个热衷的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.nature[17] < 50:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_HUMOR_MAN)
def handle_is_humor_man(character_id: int) -> int:
    """
    校验角色是否是一个幽默的人
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    value = 0
    character_data: game_type.Character = cache.character_data[character_id]
    for i in {0, 1, 2, 5, 13, 14, 15, 16}:
        nature = character_data.nature[i]
        if nature > 50:
            value -= nature - 50
        else:
            value += 50 - nature
    if value > 0:
        return 1
    return 0
