import random
from Script.Design import handle_state_machine, constant
from Script.Core import cache_control, game_type

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_state_machine.add_state_machine(constant.StateMachine.BUY_RAND_FOOD_AT_CAFETERIA)
def character_buy_rand_food_at_restaurant(character_id: int):
    """
    在取餐区购买随机食物
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    new_food_list = []
    for food_id in cache.restaurant_data:
        if not cache.restaurant_data[food_id]:
            continue
        for food_uid in cache.restaurant_data[food_id]:
            now_food: game_type.Food = cache.restaurant_data[food_id][food_uid]
            if now_food.eat:
                price = round(1 + sum(now_food.feel.values()) * now_food.quality / 100, 2)
                if price <= character_data.money:
                    new_food_list.append(food_id)
            break
    if not new_food_list:
        return
    now_food_id = random.choice(new_food_list)
    now_food = cache.restaurant_data[now_food_id][
        random.choice(list(cache.restaurant_data[now_food_id].keys()))
    ]
    price = round(1 + sum(now_food.feel.values()) * now_food.quality / 100, 2)
    character_data.food_bag[now_food.uid] = now_food
    if character_data.behavior.start_time < cache.game_time:
        character_data.behavior.start_time += 60
    del cache.restaurant_data[now_food_id][now_food.uid]
    character_data.money - price


@handle_state_machine.add_state_machine(constant.StateMachine.BUY_RAND_DRINKS_AT_CAFETERIA)
def character_buy_rand_drinks_at_restaurant(character_id: int):
    """
    在取餐区购买随机饮料
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    new_food_list = []
    for food_id in cache.restaurant_data:
        if not cache.restaurant_data[food_id]:
            continue
        for food_uid in cache.restaurant_data[food_id]:
            now_food: game_type.Food = cache.restaurant_data[food_id][food_uid]
            if now_food.eat and 28 in now_food.feel:
                price = round(1 + sum(now_food.feel.values()) * now_food.quality / 100, 2)
                if price <= character_data.money:
                    new_food_list.append(food_id)
            break
    if not new_food_list:
        return
    now_food_id = random.choice(new_food_list)
    now_food = cache.restaurant_data[now_food_id][
        random.choice(list(cache.restaurant_data[now_food_id].keys()))
    ]
    price = round(1 + sum(now_food.feel.values()) * now_food.quality / 100, 2)
    character_data.food_bag[now_food.uid] = now_food
    if character_data.behavior.start_time < cache.game_time:
        character_data.behavior.start_time += 60
    del cache.restaurant_data[now_food_id][now_food.uid]
    character_data.money - price


@handle_state_machine.add_state_machine(constant.StateMachine.BUY_GUITAR)
def character_buy_guitar(character_id: int):
    """
    购买吉他
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.item.add(4)
    if character_data.behavior.start_time < cache.game_time:
        character_data.behavior.start_time += 60
