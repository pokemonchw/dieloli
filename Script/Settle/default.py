from Script.Design import (
    settle_behavior,
    talk,
    talk_cache,
    game_time,
    map_handle,
)
from Script.Core import constant, cache_contorl, game_type,text_loading


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


@settle_behavior.add_settle_behavior(constant.Behavior.MOVE)
def settle_move(character_id: int):
    """
    结算角色移动行为
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache_contorl.character_data[character_id]
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
        eat_weight = 100
        if food.weight < eat_weight:
            eat_weight = food.weight
        for feel in food.feel:
            now_feel_value = food.feel[feel]
            now_feel_value = now_feel_value / food.weight
            now_feel_value *= eat_weight
            if feel in character_data.status["BodyFeeling"]:
                if feel in ("Hunger", "Thirsty"):
                    character_data.status["BodyFeeling"][feel] -= now_feel_value
                else:
                    character_data.status["BodyFeeling"][feel] += now_feel_value
            elif feel in character_data.status["SexFeel"]:
                character_data.status["SexFeel"][feel] += now_feel_value
            elif feel in character_data.status["PsychologicalFeeling"]:
                character_data.status["PsychologicalFeeling"][feel] += now_feel_value
        food.weight -= eat_weight
        food_name = ""
        if food.recipe == -1:
            food_config = text_loading.get_game_data(constant.FilePath.FOOD_PATH,food.id)
            food_name = food_config["Name"]
        else:
            food_name = cache_contorl.recipe_data[food.recipe].name
        character_data.behavior["FoodName"] = food_name
        character_data.behavior["FoodQuality"] = food.quality
        if food.weight <= 0:
            del character_data.food_bag[food.uid]
