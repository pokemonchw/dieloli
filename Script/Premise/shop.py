from Script.Core import game_type, cache_control
from Script.Design import handle_premise, constant

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.CAFETERIA_HAS_FOOD)
def handle_cafeteria_has_food(character_id: int) -> int:
    """
    校验食堂是否有卖吃的东西
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    for food_id in cache.restaurant_data:
        if not cache.restaurant_data[food_id]:
            continue
        for food_uid in cache.restaurant_data[food_id]:
            now_food: game_type.Food = cache.restaurant_data[food_id][food_uid]
            if now_food.eat:
                return 1
            break
    return 0


@handle_premise.add_premise(constant.Premise.CAFETERIA_HAS_DRINK)
def handle_cafeteria_has_drink(character_id: int) -> int:
    """
    校验食堂是否有卖喝的东西
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    for food_id in cache.restaurant_data:
        if not cache.restaurant_data[food_id]:
            continue
        for food_uid in cache.restaurant_data[food_id]:
            now_food: game_type.Food = cache.restaurant_data[food_id][food_uid]
            if now_food.eat and 28 in now_food.feel:
                return 1
            break
    return 0
