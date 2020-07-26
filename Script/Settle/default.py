from Script.Design import (
    settle_behavior,
    talk,
    talk_cache,
    game_time,
    map_handle,
)
from Script.Core import constant, cache_contorl, game_type


@settle_behavior.add_settle_behavior(constant.Behavior.REST)
def settle_rest(character_id: int):
    """
    结算角色休息行为
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache_contorl.character_data[character_id]
    start_time = game_time.game_time_to_datetime(
        character_data.behavior["StartTime"]
    )
    now_time = game_time.game_time_to_datetime(cache_contorl.game_time)
    add_time = int((now_time - start_time).seconds / 60)
    add_hit_point = add_time * 5
    add_mana_point = add_time * 10
    character_data.hit_point += add_hit_point
    character_data.mana_point += add_mana_point
    if character_data.hit_point > character_data.hit_point_max:
        add_hit_point -= (
            character_data.hit_point - character_data.hit_point_max
        )
        character_data.hit_point = character_data.hit_point_max
    if character_data.mana_point > character_data.mana_point_max:
        add_mana_point -= (
            character_data.mana_point - character_data.mana_point_max
        )
        character_data.mana_point = character_data.mana_point_max
    character_data.status["BodyFeeling"]["Hunger"] += add_time * 0.02
    character_data.status["BodyFeeling"]["Thirsty"] += add_time * 0.02
    cache_contorl.status_up_text.setdefault(character_id, {})
    cache_contorl.status_up_text[character_id]["HitPoint"] = add_hit_point
    cache_contorl.status_up_text[character_id]["ManaPoint"] = add_mana_point
    if character_id == cache_contorl.character_data[0].target_character_id and character_id:
        talk_cache.tg = character_data
        talk_cache.me = cache_contorl.character_data[0]
        talk.handle_talk(constant.Behavior.REST)


@settle_behavior.add_settle_behavior(constant.Behavior.MOVE)
def settle_move(character_id: int):
    """
    结算角色移动行为
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache_contorl.character_data[character_id]
    if (
        character_data.behavior["MoveTarget"]
        == cache_contorl.character_data[0].position
        or character_data.position == cache_contorl.character_data[0].position
    ):
        talk_cache.tg = character_data
        talk_cache.me = cache_contorl.character_data[0]
        scene_path_str = map_handle.get_map_system_path_str_for_list(
            character_data.behavior["MoveTarget"]
        )
        scene_data = cache_contorl.scene_data[scene_path_str]
        talk_cache.scene = scene_data["SceneName"]
        talk_cache.scene_tag = scene_data["SceneTag"]
        talk.handle_talk(constant.Behavior.MOVE)
    map_handle.character_move_scene(
        character_data.position,
        character_data.behavior["MoveTarget"],
        character_id,
    )
    character_data.status["BodyFeeling"]["Hunger"] += (
        character_data.behavior["Duration"] * 0.02
    )
    character_data.status["BodyFeeling"]["Thirsty"] += (
        character_data.behavior["Duration"] * 0.02
    )


@settle_behavior.add_settle_behavior(constant.Behavior.EAT)
def settle_eat(character_id: int):
    """
    结算角色进食行为
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache_contorl.character_data[character_id]
    if character_data.behavior["EatFood"] != None:
        food: game_type.Food = character_data.behavior["EatFood"]
        for feel in food.feel:
            if feel in character_data.status["BodyFeeling"]:
                if feel in ("Hunger", "Thirsty"):
                    character_data.status["BodyFeeling"][feel] -= food.feel[
                        feel
                    ]
                else:
                    character_data.status["BodyFeeling"][feel] += food.feel[
                        feel
                    ]
            elif feel in character_data.status["SexFeel"]:
                character_data.status["SexFeel"][feel] += food.feel[feel]
            elif feel in character_data.status["PsychologicalFeeling"]:
                character_data.status["PsychologicalFeeling"][
                    feel
                ] += food.feel[feel]
        if character_id == cache_contorl.character_data[0].target_character_id:
            talk_cache.tg = character_data
            talk_cache.me = cache_contorl.character_data[0]
            scene_path_str = map_handle.get_map_system_path_str_for_list(
                character_data.behavior["MoveTarget"]
            )
            scene_data = cache_contorl.scene_data[scene_path_str]
            talk_cache.scene = scene_data["SceneName"]
            talk_cache.scene_tag = scene_data["SceneTag"]
            talk.handle_talk(constant.Behavior.EAT)
        del character_data.food_bag[food.id]
