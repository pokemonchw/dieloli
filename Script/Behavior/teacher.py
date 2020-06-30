import copy
import random
from Script.Core import cache_contorl, constant
from Script.Design import (
    game_time,
    character_behavior,
    map_handle,
    character_move,
    character,
)


@character_behavior.add_behavior(
    "Teacher", constant.CharacterStatus.STATUS_ARDER
)
def arder_behavior(character_id: int):
    """
    教师休闲状态行为
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache_contorl.character_data[character_id]
    now_time_slice = game_time.get_now_time_slice(character_id)
    if now_time_slice["InCourse"]:
        character.init_character_behavior_start_time(character_id)
        if character_data.position != map_handle.get_map_system_path_for_str(
            character_data.classroom
        ):
            character.character_move_to_classroom(character_id)
        else:
            character.character_attend_class(character_id)
    elif now_time_slice["TimeSlice"] == constant.TimeSlice.TIME_BREAKFAST:
        if character_data.status["BodyFeeling"]["Hunger"] > 16:
            now_scene_str = map_handle.get_map_system_path_str_for_list(
                character_data.position
            )
            now_scene_data = cache_contorl.scene_data[now_scene_str]
            if now_scene_data["SceneTag"] == "Cafeteria":
                if not len(character_data.food_bag):
                    character.character_buy_rand_food_at_restaurant(character_id)
                else:
                    character.character_move_to_rand_restaurant(character_id)
            elif not len(character_data.food_bag):
                if now_time_slice["ToCourse"] >= 13:
                    character.character_move_to_rand_cafeteria(character_id)
                else:
                    if now_scene_str == character_data.classroom:
                        character.character_rest_to_time(character_id,now_time_slice["ToCourse"])
                    else:
                        character.character_move_to_classroom(character_id)
            else:
                if now_time_slice["ToCourse"] >= 13:
                    if now_scene_data["SceneTag"] == "Restaurant":
                        character.character_eat_rand_food(character_id)
                    else:
                        character.character_move_to_rand_restaurant(character_id)
                else:
                    if now_scene_str == character_data.classroom:
                        character.character_rest_to_time(character_id,now_time_slice["ToCourse"])
                    else:
                        character.character_move_to_classroom(character_id)
        else:
            if now_time_slice["ToCourse"] <= 13:
                if now_scene_str == character_data.classroom:
                    character.character_rest_to_time(character_id,now_time_slice["ToCourse"])
                else:
                    character.character_move_to_classroom(character_id)
    return 1
