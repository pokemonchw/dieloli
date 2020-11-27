from Script.Design import (
    settle_behavior,
    talk,
    talk_cache,
    game_time,
    map_handle,
)
from Script.Core import constant, cache_contorl, game_type
from Script.Config import game_config

cache:game_type.Cache = cache_contorl.cache
""" 游戏缓存数据 """

@settle_behavior.add_settle_behavior(constant.Behavior.REST)
def settle_rest(character_id: int):
    """
    结算角色休息行为
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache.character_data[character_id]
    start_time = character_data.behavior.start_time
    now_time = cache.game_time
    add_time = int((now_time - start_time).seconds / 60)
    add_hit_point = add_time * 5
    add_mana_point = add_time * 10
    character_data.hit_point += add_hit_point
    character_data.mana_point += add_mana_point
    if character_data.hit_point > character_data.hit_point_max:
        add_hit_point -= character_data.hit_point - character_data.hit_point_max
        character_data.hit_point = character_data.hit_point_max
    if character_data.mana_point > character_data.mana_point_max:
        add_mana_point -= character_data.mana_point - character_data.mana_point_max
        character_data.mana_point = character_data.mana_point_max
    cache.status_up_text.setdefault(character_id, {})
    cache.status_up_text[character_id]["HitPoint"] = add_hit_point
    cache.status_up_text[character_id]["ManaPoint"] = add_mana_point


@settle_behavior.add_settle_behavior(constant.Behavior.MOVE)
def settle_move(character_id: int):
    """
    结算角色移动行为
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache.character_data[character_id]
    map_handle.character_move_scene(
        character_data.position,
        character_data.behavior.move_target,
        character_id,
    )


@settle_behavior.add_settle_behavior(constant.Behavior.EAT)
def settle_eat(character_id: int):
    """
    结算角色进食行为
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache.character_data[character_id]
    if character_data.behavior.eat_food != None:
        food: game_type.Food = character_data.behavior.eat_food
        eat_weight = 100
        if food.weight < eat_weight:
            eat_weight = food.weight
        for feel in food.feel:
            now_feel_value = food.feel[feel]
            now_feel_value = now_feel_value / food.weight
            now_feel_value *= eat_weight
            if feel in character_data.status:
                if feel in {27, 28}:
                    character_data.status[feel] -= now_feel_value
                    if character_data.status[feel] < 0:
                        character_data.status[feel] = 0
                else:
                    character_data.status[feel] += now_feel_value
        food.weight -= eat_weight
        food_name = ""
        if food.recipe == -1:
            food_config = game_config.config_food[food.id]
            food_name = food_config.name
        else:
            food_name = cache.recipe_data[food.recipe].name
        character_data.behavior.food_name = food_name
        character_data.behavior.food_quality = food.quality
        if food.weight <= 0:
            del character_data.food_bag[food.uid]
