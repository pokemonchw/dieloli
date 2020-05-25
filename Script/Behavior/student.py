from Script.Core import cache_contorl, constant
from Script.Design import (
    game_time,
    settle_behavior,
    character_behavior,
    character_move,
    map_handle,
)


@character_behavior.add_behavior(
    "Student", constant.CharacterStatus.STATUS_ARDER
)
def arder_behavior(character_id: int):
    """
    休闲状态行为
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache_contorl.character_data[character_id]
    now_time_slice = game_time.get_now_time_slice(character_id)
    if now_time_slice["InCourse"]:
        character_data.behavior["StartTime"] = cache_contorl.game_time
        if character_data.position != map_handle.get_map_system_path_for_str(
            character_data.classroom
        ):
            _, _, move_path, move_time = character_move.character_move(
                character_id,
                map_handle.get_map_system_path_for_str(
                    character_data.classroom
                ),
            )
            character_data.behavior["BehaviorId"] = constant.Behavior.MOVE
            character_data.behavior["MoveTarget"] = move_path
            character_data.behavior["Duration"] = move_time
            character_data.state = constant.CharacterStatus.STATUS_MOVE
        else:
            character_data.behavior[
                "BehaviorId"
            ] = constant.Behavior.ATTEND_CLASS
            character_data.behavior["Duration"] = now_time_slice["EndCourse"]
            character_data.behavior["MoveTarget"] = []
            character_data.state = constant.CharacterStatus.STATUS_ATTEND_CLASS
    elif now_time_slice["TimeSlice"] == constant.TimeSlice.TIME_BREAKFAST:
        pass


@character_behavior.add_behavior(
    "Student", constant.CharacterStatus.STATUS_REST
)
def rest_behavior(character_id: int):
    """
    休息状态行为
    Keyword arguments:
    character_id -- 角色id
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
        character_data.behavior["StartTime"] = now_time


@character_behavior.add_behavior(
    "Student", constant.CharacterStatus.STATUS_ATTEND_CLASS
)
def attend_class(character_id: int):
    """
    上课状态行为
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache_contorl.character_data[character_id]
