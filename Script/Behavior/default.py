from Script.Design import game_time, character_behavior, settle_behavior, character
from Script.Core import constant, cache_contorl


@character_behavior.add_behavior(
    "Default", constant.CharacterStatus.STATUS_ARDER
)
def arder_behavior(character_id: int) -> bool:
    """
    休闲状态行为
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    bool -- 当前npc已结束ai行为判断校验
    """
    character_data = cache_contorl.character_data[character_id]
    now_time_slice = game_time.get_now_time_slice(character_id)
    return 1


@character_behavior.add_behavior(
    "Default", constant.CharacterStatus.STATUS_REST
)
def rest_behavior(character_id: int) -> bool:
    """
    休息状态行为
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    bool -- 当前npc已结束ai行为判断校验
    """
    character_data = cache_contorl.character_data[character_id]
    start_time = character_data.behavior["StartTime"]
    end_time = game_time.datetime_to_game_time(
        game_time.get_sub_date(
            minute=character_data.behavior["Duration"],
            old_date=game_time.game_time_to_datetime(start_time),
        )
    )
    now_time = cache_contorl.game_time
    time_judge = game_time.judge_date_big_or_small(now_time, end_time)
    if time_judge:
        settle_behavior.handle_settle_behavior(character_id)
        character_data.behavior["BehaviorId"] = constant.Behavior.SHARE_BLANKLY
        character_data.state = constant.CharacterStatus.STATUS_ARDER
    if time_judge == 1:
        character_data.behavior["StartTime"] = end_time
    elif time_judge == 2:
        character.init_character_behavior_start_time(character_id)
        return 1
    else:
        return 1


@character_behavior.add_behavior(
    "Default", constant.CharacterStatus.STATUS_MOVE
)
def move_behavior(character_id: int) -> bool:
    """
    移动状态行为
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    bool -- 当前npc已结束ai行为判断校验
    """
    character_data = cache_contorl.character_data[character_id]
    now_time = game_time.game_time_to_datetime(cache_contorl.game_time)
    end_time = game_time.get_sub_date(
        character_data.behavior["Duration"], old_date=start_time,
    )
    if character_data.position == character_data.behavior["MoveTarget"]:
        character_data.behavior["StartTime"] = game_time.datetime_to_game_time(
            end_time
        )
        character_data.behavior["Duration"] = 0
        character_data.behavior[
            "BehaviorId"
        ] = constant.CharacterStatus.STATUS_ARDER
        character_data.state = constant.CharacterStatus.STATUS_ARDER
        if end_time == now_time:
            return 1
    else:
        start_time = game_time.game_time_to_datetime(
            character_data.behavior["StartTime"]
        )
        now_time = game_time.game_time_to_datetime(cache_contorl.game_time)
        end_time = game_time.get_sub_date(
            character_data.behavior["Duration"], old_date=start_time,
        )
        time_judge = game_time.judge_date_big_or_small(
            game_time.datetime_to_game_time(end_time), cache_contorl.game_time
        )
        if time_judge == 0 or time_judge == 2:
            settle_behavior.handle_settle_behavior(character_id)
            character_data.behavior[
                "StartTime"
            ] = game_time.datetime_to_game_time(end_time)
            character_data.behavior["Duration"] = 0
            character_data.behavior[
                "BehaviorId"
            ] = constant.Behavior.SHARE_BLANKLY
            character_data.state = constant.CharacterStatus.STATUS_ARDER
        if time_judge == 1 or time_judge == 2:
            return 1
    return 0
