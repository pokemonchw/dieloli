from Script.Design import map_handle, character_move, game_time
from Script.Core import cache_contorl, constant


def move_action(character_id: int, target_path: list):
    """
    移动行为执行
    Keyword arguments:
    character_id -- 角色id
    target_path -- 目标地点
    """
    character_data = cache_contorl.character_data["character"][character_id]
    character_data.behavior[
        "BehaviorId"
    ] = constant.CharacterStatus.STATUS_MOVE
    if character_data.position == target_path:
        character_data.behavior["StartTime"] = 0
        character_data.behavior["Duration"] = 0
        character_data.behavior[
            "BehaviorId"
        ] = constant.CharacterStatus.STATUS_ARDER
    else:
        if character_data.behavior["MoveTarget"] != []:
            end_time = game_time.get_sub_date(
                character_data.behavior["Duration"],
                0,
                0,
                0,
                0,
                character_data.behavior["StartTime"],
            )
            if game_time.judge_date_big_or_small(
                end_time, cache_contorl.game_time
            ):
                pass
        (
            move_status,
            _,
            now_target_position,
            now_need_time,
        ) = character_move.character_move(character_id, target_path)
        if move_status == "Null":
            pass
        else:
            pass
