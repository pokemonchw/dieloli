import copy
from Script.Core import cache_contorl, constant
from Script.Design import (
    game_time,
    settle_behavior,
    character_behavior,
    character_move,
    map_handle,
    character,
)


@character_behavior.add_behavior(
    "Student", constant.CharacterStatus.STATUS_ARDER
)
def arder_behavior(character_id: int) -> bool:
    """
    学生休闲状态行为
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    bool -- 当前npc已结束ai行为判断校验
    """
    character_data = cache_contorl.character_data[character_id]
    now_time_slice = game_time.get_now_time_slice(character_id)
    if now_time_slice["InCourse"]:
        character.init_character_behavior_start_time(character_id)
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
    return 1


@character_behavior.add_behavior(
    "Student", constant.CharacterStatus.STATUS_ATTEND_CLASS
)
def attend_class(character_id: int) -> bool:
    """
    上课状态行为
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    bool -- 当前npc已结束ai行为判断校验
    """
    character_data = cache_contorl.character_data[character_id]
    return 1
