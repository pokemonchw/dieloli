from types import FunctionType
from Script.Design import (
    settle_behavior,
    constant,
)
from Script.Core import (
    game_type,
    cache_control,
    get_text,
)
from Script.UI.Model import draw
from Script.Config import normal_config


_: FunctionType = get_text._
""" 翻译api """
window_width: int = normal_config.config_normal.text_width
""" 窗体宽度 """
cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.WEAR_UNDERWEAR)
def handle_wear_underwear(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int
):
    """
    穿上上衣
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 1 in character_data.clothing:
        value_dict = {}
        for clothing in character_data.clothing[1]:
            clothing_data: game_type.Clothing = character_data.clothing[1][clothing]
            value_dict[clothing_data.cleanliness] = clothing
        now_value = max(value_dict.keys())
        character_data.put_on[1] = value_dict[now_value]
        now_clothing = character_data.clothing[1][value_dict[now_value]]
        change_data.wear[1] = now_clothing


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.WEAR_UNDERPANTS)
def handle_wear_underpants(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int
):
    """
    穿上内裤
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 7 in character_data.clothing:
        value_dict = {}
        for clothing in character_data.clothing[7]:
            clothing_data: game_type.Clothing = character_data.clothing[7][clothing]
            value_dict[clothing_data.cleanliness] = clothing
        now_value = max(value_dict.keys())
        character_data.put_on[7] = value_dict[now_value]
        now_clothing = character_data.clothing[7][value_dict[now_value]]
        change_data.wear[7] = now_clothing


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.WEAR_BRA)
def handle_wear_bra(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int
):
    """
    穿上胸罩
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 6 in character_data.clothing:
        value_dict = {}
        for clothing in character_data.clothing[6]:
            clothing_data: game_type.Clothing = character_data.clothing[6][clothing]
            value_dict[clothing_data.cleanliness] = clothing
        now_value = max(value_dict.keys())
        character_data.put_on[6] = value_dict[now_value]
        now_clothing = character_data.clothing[6][value_dict[now_value]]
        change_data.wear[6] = now_clothing


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.WEAR_PANTS)
def handle_wear_pants(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int
):
    """
    穿上裤子
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 2 in character_data.clothing:
        value_dict = {}
        for clothing in character_data.clothing[2]:
            clothing_data: game_type.Clothing = character_data.clothing[2][clothing]
            value_dict[clothing_data.cleanliness] = clothing
        now_value = max(value_dict.keys())
        character_data.put_on[2] = value_dict[now_value]
        now_clothing = character_data.clothing[2][value_dict[now_value]]
        change_data.wear[2] = now_clothing


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.WEAR_SKIRT)
def handle_wear_skirt(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int
):
    """
    穿上短裙
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 3 in character_data.clothing:
        value_dict = {}
        for clothing in character_data.clothing[3]:
            clothing_data: game_type.Clothing = character_data.clothing[3][clothing]
            value_dict[clothing_data.cleanliness] = clothing
        now_value = max(value_dict.keys())
        character_data.put_on[3] = value_dict[now_value]
        now_clothing = character_data.clothing[3][value_dict[now_value]]
        change_data.wear[3] = now_clothing


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.WEAR_SHOES)
def handle_wear_shoes(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int
):
    """
    穿上鞋子
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 4 in character_data.clothing:
        value_dict = {}
        for clothing in character_data.clothing[4]:
            clothing_data: game_type.Clothing = character_data.clothing[4][clothing]
            value_dict[clothing_data.cleanliness] = clothing
        now_value = max(value_dict.keys())
        character_data.put_on[4] = value_dict[now_value]
        now_clothing = character_data.clothing[4][value_dict[now_value]]
        change_data.wear[4] = now_clothing


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.WEAR_SOCKS)
def handle_wear_socks(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int
):
    """
    穿上袜子
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 5 in character_data.clothing:
        value_dict = {}
        for clothing in character_data.clothing[5]:
            clothing_data: game_type.Clothing = character_data.clothing[5][clothing]
            value_dict[clothing_data.cleanliness] = clothing
        now_value = max(value_dict.keys())
        character_data.put_on[5] = value_dict[now_value]
        now_clothing = character_data.clothing[5][value_dict[now_value]]
        change_data.wear[5] = now_clothing


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.WEAR_COATS)
def handle_wear_coat(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int
):
    """
    穿上外套
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 0 in character_data.clothing:
        value_dict = {}
        for clothing in character_data.clothing[0]:
            clothing_data: game_type.Clothing = character_data.clothing[0][clothing]
            value_dict[clothing_data.cleanliness] = clothing
        now_value = max(value_dict.keys())
        character_data.put_on[0] = value_dict[now_value]
        now_clothing = character_data.clothing[0][value_dict[now_value]]
        change_data.wear[0] = now_clothing


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.UNDRESS_UNDERWEAR)
def handle_undress_underwear(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int
):
    """
    脱下上衣
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 1 in character_data.put_on and character_data.put_on[1] not in {"",None}:
        now_clothing_id = character_data.put_on[1]
        now_clothing = character_data.clothing[1][now_clothing_id]
        change_data.undress[1] = now_clothing
    character_data.put_on[1] = ""


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.UNDRESS_UNDERPANTS)
def handle_undress_underpants(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int
):
    """
    脱下内裤
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 7 in character_data.put_on and character_data.put_on[7] not in {"",None}:
        now_clothing_id = character_data.put_on[7]
        now_clothing = character_data.clothing[7][now_clothing_id]
        change_data.undress[7] = now_clothing
    character_data.put_on[7] = ""


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.UNDRESS_BRA)
def handle_undress_bra(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int
):
    """
    脱下胸罩
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 6 in character_data.put_on and character_data.put_on[6] not in {"",None}:
        now_clothing_id = character_data.put_on[6]
        now_clothing = character_data.clothing[6][now_clothing_id]
        change_data.undress[6] = now_clothing
    character_data.put_on[6] = ""


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.UNDRESS_PANTS)
def handle_undress_pants(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int
):
    """
    脱下裤子
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 2 in character_data.put_on and character_data.put_on[2] not in {"",None}:
        now_clothing_id = character_data.put_on[2]
        now_clothing = character_data.clothing[2][now_clothing_id]
        change_data.undress[2] = now_clothing
    character_data.put_on[2] = ""


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.UNDRESS_SKIRT)
def handle_undress_skirt(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int
):
    """
    脱下裙子
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 3 in character_data.put_on and character_data.put_on[3] not in {"",None}:
        now_clothing_id = character_data.put_on[3]
        now_clothing = character_data.clothing[3][now_clothing_id]
        change_data.undress[3] = now_clothing
    character_data.put_on[3] = ""


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.UNDRESS_SHOES)
def handle_undress_shoes(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int
):
    """
    脱下鞋子
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 4 in character_data.put_on and character_data.put_on[4] not in {"",None}:
        now_clothing_id = character_data.put_on[4]
        now_clothing = character_data.clothing[4][now_clothing_id]
        change_data.undress[4] = now_clothing
    character_data.put_on[4] = ""


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.UNDRESS_SOCKS)
def handle_undress_socks(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int
):
    """
    脱下袜子
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 5 in character_data.put_on and character_data.put_on[5] not in {"",None}:
        now_clothing_id = character_data.put_on[5]
        now_clothing = character_data.clothing[5][now_clothing_id]
        change_data.undress[5] = now_clothing
    character_data.put_on[5] = ""


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.UNDRESS_COAT)
def handle_undress_coat(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int
):
    """
    脱下外套
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 0 in character_data.put_on and character_data.put_on[0] not in {"",None}:
        now_clothing_id = character_data.put_on[0]
        now_clothing = character_data.clothing[0][now_clothing_id]
        change_data.undress[0] = now_clothing
    character_data.put_on[0] = ""


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.UNDRESS_TARGET_UNDERWEAR)
def handle_undress_target_underwear(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int
):
    """
    脱下交互对象的上衣
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.cid == character_id:
        return
    if 1 in target_data.put_on and target_data.put_on[1] not in {"",None}:
        now_clothing_id = target_data.put_on[1]
        now_clothing = target_data.clothing[1][now_clothing_id]
        change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
        target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
        target_change.undress[1] = now_clothing
    target_data.put_on[1] = ""


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.UNDRESS_TARGET_UNDERPANTS)
def handle_undress_target_underpants(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int
):
    """
    脱下交互对象的内裤
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.cid == character_id:
        return
    if 7 in target_data.put_on and target_data.put_on[7] not in {"",None}:
        now_clothing_id = target_data.put_on[7]
        now_clothing = target_data.clothing[7][now_clothing_id]
        change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
        target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
        target_change.undress[7] = now_clothing
    target_data.put_on[7] = ""


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.UNDRESS_TARGET_BRA)
def handle_undress_target_bra(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int
):
    """
    脱下交互对象的胸罩
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.cid == character_id:
        return
    if 6 in target_data.put_on and target_data.put_on[6] not in {"",None}:
        now_clothing_id = target_data.put_on[6]
        now_clothing = target_data.clothing[6][now_clothing_id]
        change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
        target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
        target_change.undress[6] = now_clothing
    target_data.put_on[6] = ""


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.UNDRESS_TARGET_PANTS)
def handle_undress_target_pants(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int
):
    """
    脱下交互对象的裤子
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.cid == character_id:
        return
    if 2 in target_data.put_on and target_data.put_on[2] not in {"",None}:
        now_clothing_id = target_data.put_on[2]
        now_clothing = target_data.clothing[2][now_clothing_id]
        change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
        target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
        target_change.undress[2] = now_clothing
    target_data.put_on[2] = ""


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.UNDRESS_TARGET_SKIRT)
def handle_undress_target_skirt(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int
):
    """
    脱下交互对象的裙子
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.cid == character_id:
        return
    if 3 in target_data.put_on and target_data.put_on[3] not in {"",None}:
        now_clothing_id = target_data.put_on[3]
        now_clothing = target_data.clothing[3][now_clothing_id]
        change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
        target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
        target_change.undress[3] = now_clothing
    target_data.put_on[3] = ""


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.UNDRESS_TARGET_SHOES)
def handle_undress_target_shoes(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int
):
    """
    脱下交互对象的鞋子
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.cid == character_id:
        return
    if 4 in target_data.put_on and target_data.put_on[4] not in {"",None}:
        now_clothing_id = target_data.put_on[4]
        now_clothing = target_data.clothing[4][now_clothing_id]
        change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
        target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
        target_change.undress[4] = now_clothing
    target_data.put_on[4] = ""


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.UNDRESS_TARGET_SOCKS)
def handle_undress_target_socks(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int
):
    """
    脱下交互对象的袜子
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.cid == character_id:
        return
    if 5 in target_data.put_on and target_data.put_on[5] not in {"",None}:
        now_clothing_id = target_data.put_on[5]
        now_clothing = target_data.clothing[5][now_clothing_id]
        change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
        target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
        target_change.undress[5] = now_clothing
    target_data.put_on[5] = ""


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.UNDRESS_TARGET_COATS)
def handle_undress_target_coats(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int
):
    """
    脱下交互对象的外套
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.cid == character_id:
        return
    if 0 in target_data.put_on and target_data.put_on[0] not in {"",None}:
        now_clothing_id = target_data.put_on[0]
        now_clothing = target_data.clothing[0][now_clothing_id]
        change_data.target_change.setdefault(target_data.cid, game_type.TargetChange())
        target_change: game_type.TargetChange = change_data.target_change[target_data.cid]
        target_change.undress[0] = now_clothing
    target_data.put_on[0] = ""
