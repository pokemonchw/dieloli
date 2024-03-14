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
from Script.UI.Moudle import draw
from Script.Config import normal_config


_: FunctionType = get_text._
""" 翻译api """
window_width: int = normal_config.config_normal.text_width
""" 窗体宽度 """
cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.TARGET_FOLLOW_SELF)
def handle_target_follow_self(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int
):
    """
    让交互对象跟随自己
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if not character_data.target_character_id:
        return
    if character_data.target_character_id == character_id:
        return
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.dead:
        return
    if target_data.position != character_data.position:
        return
    target_data.follow = character_id
    character_data.pulling = target_data.cid


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.UNFOLLOW)
def handle_unfollow(
    character_id: int, add_time: int, change_data: game_type.CharacterStatusChange, now_time: int
):
    """
    取消跟随
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.follow != -1:
        target_data: game_type.Character = cache.character_data[character_data.pulling]
        if target_data.pulling == character_id:
            target_data.pulling = -1
    character_data.follow = -1


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.FIRST_KISS)
def handle_first_kiss(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    记录初吻
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.social_contact_data.setdefault(character_id, 0)
    if character_data.first_kiss == -1:
        character_data.first_kiss = target_data.cid
        character_data.behavior.temporary_status.lose_first_kiss = 1
        if (not character_id) or (not target_data.cid):
            now_draw = draw.NormalDraw()
            now_draw.text = _("{character_name}失去了初吻\n").format(
                character_name=character_data.name
            )
            now_draw.width = window_width
            now_draw.draw()
    if target_data.first_kiss == -1:
        target_data.first_kiss = character_id
        target_data.behavior.temporary_status.lose_first_kiss = 1
        if (not character_id) or (not target_data.cid):
            now_draw = draw.NormalDraw()
            now_draw.text = _("{character_name}失去了初吻\n").format(character_name=target_data.name)
            now_draw.width = window_width
            now_draw.draw()


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.FIRST_HAND_IN_HAND)
def handle_first_hand_in_hand(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    记录初次牵手
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    if not add_time:
        return
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.social_contact_data.setdefault(character_id, 0)
    if character_data.first_hand_in_hand == -1:
        character_data.first_kiss = target_data.cid
    if target_data.first_hand_in_hand == -1:
        target_data.first_kiss = character_id
