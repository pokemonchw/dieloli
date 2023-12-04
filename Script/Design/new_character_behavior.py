from typing import List
from joblib import Parallel, delayed
from Script.Core import (
    cache_control,
    game_type
)
from Script.Design import (
    character_handle,
    constant,
    game_time
)

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


def init_character_behavior():
    """
    角色行为树总控制
    """
    character_handle.build_similar_character_searcher()
    cache.character_target_data = {}
    cache.character_target_score_data = {}
    while 1:
        if len(cache.over_behavior_character) >= len(cache.character_data):
            break


def character_classify_by_behavior_state(character_list: List[int]) -> (List[int], List[int], List[int]):
    """
    按角色的行为树状态对角色进行分组
    Keyword arguments:
    character_list -- 要分组的角色id列表
    Return arguments:
    List[int] -- 需要查找行动目标的角色列表
    List[int] -- 状态机未结束的角色列表
    List[int] -- 需要进行状态结算的角色列表
    """
    find_target_character_list, run_character_list, completed_character_list = determine_character_behavior_state(character_list)


def determine_character_behavior_state(character_id: int) -> int:
    """
    判断角色的行为树状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    str -- 1:寻找目标 2:状态进行 3:状态结算
    """
    character_data = cache.character_data[character_id]
    if character_data.state == constant.CharacterStatus.STATUS_ARDER:
        return 1
    start_time = character_data.behavior.start_time
    end_time = start_time + 60 * character_data.behavior.duration
    time_judge = game_time.judge_date_big_or_small(cache.game_time,end_time)
    if time_judge == 0:
        return 2
    return 3
