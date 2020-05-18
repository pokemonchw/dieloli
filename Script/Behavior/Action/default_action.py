from Script.Design import map_handle, character_move, game_time
from Script.Core import cache_contorl, constant


def move_action(character_id: int, target_path: list):
    """
    移动行为执行
    Keyword arguments:
    character_id -- 角色id
    target_path -- 目标地点
    """
    character_data = cache_contorl.character_data[character_id]
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
                game_time.game_time_to_datetime(
                    character_data.behavior["StartTime"]
                ),
            )
            time_judge = game_time.judge_date_big_or_small(
                game_time.datetime_to_game_time(end_time),
                cache_contorl.game_time,
            )
            if time_judge == 0:
                map_handle.character_move_scene(
                    character_data.position,
                    character_data.behavior["MoveTarget"],
                    character_id,
                )
                character_data.behavior["StartTime"] = game_time.datetime_to_game_tim(end_time)
            elif time_judge == 1:
                need_time = (
                    game_time.timetuple_to_datetime(end_time)
                    - game_time.game_time_to_time_tuple(
                        cache_contorl.game_time
                    )
                ).minutes
                character_data.behavior["Duration"] = need_time
                character_data.behavior[
                    "BehaviorId"
                ] = constant.Behavior.SHARE_BLANKLY
                character_data.state = constant.CharacterStatus.STATUS_ARDER
            else:
                map_handle.character_move_scene(
                    character_data.position,
                    character_data.behavior["MoveTarget"],
                    character_id,
                )
                character_data.behavior["StartTime"] = game_time.datetime_to_game_time(end_time)
        else:
            (
                move_status,
                _,
                now_target_position,
                now_need_time,
            ) = character_move.character_move(character_id, target_path)
            if move_status != "Null":
                character_data.behavior["MoveTarget"] = now_target_position
                character_data.behavior["StartTime"] = cache_contorl.game_time
                character_data.behavior["Duration"] = now_need_time
            else:
                character_data.behavior["StartTime"] = cache_contorl.game_time
                character_data.state = constant.CharacterStatus.STATUS_ARDER
                character_data.behavior[
                    "BehaviorId"
                ] = constant.Behavior.SHARE_BLANKLY
    cache_contorl.character_data[character_id] = character_data
