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


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_LIKE_PREFERENCE)
def handle_add_like_preference(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int
):
    """
    增加角色对目标的偏好倾向分
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
    now_target_id = change_data.now_target_id
    if now_target_id in character_data.dislike_preference_data:
        character_data.dislike_preference_data[now_target_id] -= 1
        if character_data.dislike_preference_data[now_target_id] <= 0:
            del character_data.dislike_preference_data[now_target_id]
    else:
        now_data = {k: character_data.like_preference_data[k] for k in character_data.like_preference_data if k != now_target_id}
        if now_data:
            if sum(character_data.like_preference_data.values()) >= 100:
                min_target = min(now_data, now_data.get)
                character_data.like_preference_data[min_target] -= 1
        character_data.like_preference_data.setdefault(now_target_id, 0)
        character_data.like_preference_data[now_target_id] += 1


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.ADD_DISLIKE_PREFERENCE)
def handle_add_dislike_preference(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int
):
    """
    增加角色对目标的厌恶倾向分
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
    now_target_id = change_data.now_target_id
    if now_target_id in character_data.like_preference_data:
        character_data.like_preference_data[now_target_id] -= 1
        if character_data.like_preference_data[now_target_id] <= 0:
            del character_data.like_preference_data[now_target_id]
    else:
        now_data = {k: character_data.dislike_preference_data[k] for k in character_data.dislike_preference_data if k != now_target_id}
        if now_data:
            if sum(character_data.dislike_preference_data.values()) >= 100:
                min_target = min(now_data, now_data.get)
                character_data.dislike_preference_data[min_target] -= 1
        character_data.dislike_preference_data.setdefault(now_target_id, 0)
        character_data.dislike_preference_data[now_target_id] += 1
