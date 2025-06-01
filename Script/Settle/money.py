from types import FunctionType
from Script.Design import (
    settle_behavior,
    map_handle,
    cooking,
    game_time,
    character,
    constant,
)
from Script.Core import (
    get_text,
    game_type,
    cache_control,
)
from Script.Config import game_config, normal_config
from Script.UI.Model import draw

_: FunctionType = get_text._
""" 翻译api """
window_width: int = normal_config.config_normal.text_width
""" 窗体宽度 """
cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_SMALL_MONEY)
def handle_add_small_money(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    增加少量金钱
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
    add_money = add_time * 0.1
    character_data.money += add_money
    change_data.money += add_money


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_MEDIUM_MONEY)
def handle_add_medium_money(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    增加中量金钱
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
    character_data.money += add_time
    change_data.money += add_time


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LARGE_MONEY)
def handle_add_large_money(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    增加大量金钱
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
    add_money = add_time * 10
    character_data.money += add_money
    change_data.money += add_money


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_SMALL_MONEY)
def handle_sub_small_money(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    减少少量金钱
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
    sub_money = add_time * 0.1
    character_data.money -= sub_money
    change_data.money -= sub_money


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_MEDIUM_MONEY)
def handle_sub_medium_money(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    减少中量金钱
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
    character_data.money -= add_time
    change_data.money -= add_time


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.SUB_LARGE_MONEY)
def handle_sub_large_money(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    减少大量金钱
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
    sub_money = add_time * 10
    character_data.money -= sub_money
    change_data.money -= add_money


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.GIFT_TARGET_SMALL_MONEY)
def handle_gift_target_small_money(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    赠送交互对象少量金钱
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
    gift_money = min(add_time*0.1,character_data.money)
    character_data.money -= gift_money
    change_data.money -= gift_money
    target_character_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_character_data.money += gift_money


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.GIFT_TARGET_MEDIUM_MONEY)
def handle_gift_target_medium_money(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    赠送交互对象中量金钱
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
    gift_money = min(add_time,character_data.money)
    character_data.money -= gift_money
    change_data.money -= gift_money
    target_character_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_character_data.money += gift_money


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.GIFT_TARGET_LARGE_MONEY)
def handle_gift_target_large_money(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    赠送交互对象大量金钱
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
    gift_money = min(add_time*10,character_data.money)
    character_data.money -= gift_money
    change_data.money -= gift_money
    target_character_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_character_data.money += gift_money


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_GIFT_SMALL_MONEY)
def handle_target_gift_small_money(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    交互对象赠送少量金钱
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
    target_character_data: game_type.Character = cache.character_data[character_data.target_character_id]
    gift_money = min(add_time*0.1, target_character_data.money)
    target_character_data.money -= gift_money
    character_data.money += gift_money
    change_data.money += gift_money


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_GIFT_MEDIUM_MONEY)
def handle_target_gift_medium_money(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    交互对象赠送中量金钱
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
    target_character_data: game_type.Character = cache.character_data[character_data.target_character_id]
    gift_money = min(add_time, target_character_data.money)
    target_character_data.money -= gift_money
    character_data.money += gift_money
    change_data.money += gift_money


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_GIFT_LARGE_MONEY)
def handle_target_gift_large_money(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    交互对象赠送大量金钱
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
    target_character_data: game_type.Character = cache.character_data[character_data.target_character_id]
    gift_money = min(add_time * 10, target_character_data.money)
    target_character_data.money -= gift_money
    character_data.money += gift_money
    change_data.money += gift_money

