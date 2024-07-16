from uuid import UUID
from Script.Design import handle_premise, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.HAVE_SMALL_MONEY)
def handle_have_small_money(character_id: int) -> int:
    """
    校验角色是否拥有少量金钱
    Keyword arguments:
    character_id -- 角色id
    Return argument:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    if character_data.target_character_id >= 1000:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.HAVE_MEDIUM_MONEY)
def handle_have_medium_money(character_id: int) -> int:
    """
    校验角色是否拥有中量金钱
    Keyword arguments:
    character_id -- 角色id
    Return argument:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    if character_data.target_character_id >= 10000:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.HAVE_LARGE_MONEY)
def handle_have_large_money(character_id: int) -> int:
    """
    校验角色是否拥有大量金钱
    Keyword arguments:
    character_id -- 角色id
    Return argument:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    if character_data.target_character_id >= 100000:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.MONEY_ENOUGH_BUY_CHEAPEST_FOOD)
def handle_money_enough_buy_cheapest_food(character_id: int) -> int:
    """
    校验是否拥有足够买最便宜的食物的钱
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    food_price = 0
    for food_id in cache.restaurant_data:
        if not cache.restaurant_data[food_id]:
            continue
        for food_uid in cache.restaurant_data[food_id]:
            now_food: game_type.Food = cache.restaurant_data[food_id][food_uid]
            if now_food.eat:
                price = round(1 + sum(now_food.feel.values()) * now_food.quality / 100, 2)
                if food_price == 0:
                    food_price = price
                else:
                    food_price = min(food_price, price)
    if food_price == 0:
        return 0
    if character_data.money >= food_price:
        return 1
    return food_price


@handle_premise.add_premise(constant.Premise.MONEY_ENOUGH_BUY_CHEAPEST_DRINK)
def handle_money_enough_buy_cheapest_drink(character_id: int) -> int:
    """
    校验是否拥有足够买最便宜的饮料的钱
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    food_price = 0
    for food_id in cache.restaurant_data:
        if not cache.restaurant_data[food_id]:
            continue
        for food_uid in cache.restaurant_data[food_id]:
            now_food: game_type.Food = cache.restaurant_data[food_id][food_uid]
            if now_food.eat and 28 in now_food.feel:
                price = round(1 + sum(now_food.feel.values()) * now_food.quality / 100, 2)
                if food_price == 0:
                    food_price = price
                else:
                    food_price = min(food_price, price)
    if food_price == 0:
        return 0
    if character_data.money >= food_price:
        return 1
    return food_price


@handle_premise.add_premise(constant.Premise.MONEY_ENOUGH_BUY_MOST_EXPENSIVE_FOOD)
def handle_money_enough_buy_most_expensive_food(character_id: int) -> int:
    """
    校验是否拥有足够买最贵的食物的钱
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    food_price = 0
    for food_id in cache.restaurant_data:
        if not cache.restaurant_data[food_id]:
            continue
        for food_uid in cache.restaurant_data[food_id]:
            now_food: game_type.Food = cache.restaurant_data[food_id][food_uid]
            if now_food.eat:
                price = round(1 + sum(now_food.feel.values()) * now_food.quality / 100, 2)
                if food_price == 0:
                    food_price = price
                else:
                    food_price = max(food_price, price)
    if food_price == 0:
        return 0
    if character_data.money >= food_price:
        return 1
    return food_price


@handle_premise.add_premise(constant.Premise.MONEY_ENOUGH_BUY_MOST_EXPENSIVE_DRINK)
def handle_money_enough_buy_most_expensive_drink(character_id: int) -> int:
    """
    校验是否拥有足够买最贵的饮料的钱
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    food_price = 0
    for food_id in cache.restaurant_data:
        if not cache.restaurant_data[food_id]:
            continue
        for food_uid in cache.restaurant_data[food_id]:
            now_food: game_type.Food = cache.restaurant_data[food_id][food_uid]
            if now_food.eat and 28 in now_food.feel:
                price = round(1 + sum(now_food.feel.values()) * now_food.quality / 100, 2)
                if food_price == 0:
                    food_price = price
                else:
                    food_price = max(food_price, price)
    if food_price == 0:
        return 0
    if character_data.money >= food_price:
        return 1
    return food_price

