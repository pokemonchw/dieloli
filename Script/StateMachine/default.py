import random
from Script.Design import handle_state_machine, character, character_move, map_handle
from Script.Core import cache_control, game_type, constant

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_state_machine.add_state_machine(constant.StateMachine.MOVE_TO_CLASS)
def character_move_to_classroom(character_id: int):
    """
    移动至所属教室
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache.character_data[character_id]
    _, _, move_path, move_time = character_move.character_move(
        character_id,
        map_handle.get_map_system_path_for_str(character_data.classroom),
    )
    character_data.behavior["BehaviorId"] = constant.Behavior.MOVE
    character_data.behavior["MoveTarget"] = move_path
    character_data.behavior["Duration"] = move_time
    character_data.state = constant.CharacterStatus.STATUS_MOVE


@handle_state_machine.add_state_machine(constant.StateMachine.MOVE_TO_RAND_RESTAURANT)
def character_move_to_rand_cafeteria(character_id: int):
    """
    移动至随机取餐区
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache.character_data[character_id]
    to_cafeteria = map_handle.get_map_system_path_for_str(random.choice(cache.place_data["Cafeteria"]))
    _, _, move_path, move_time = character_move.character_move(character_id, to_cafeteria)
    character_data.behavior["BehaviorId"] = constant.Behavior.MOVE
    character_data.behavior["MoveTarget"] = move_path
    character_data.behavior["Duration"] = move_time
    character_data.state = constant.CharacterStatus.STATUS_MOVE


@handle_state_machine.add_state_machine(constant.StateMachine.BUY_RAND_FOOD_AT_CAFETERIA)
def character_buy_rand_food_at_restaurant(character_id: int):
    """
    在取餐区购买随机食物
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache.character_data[character_id]
    food_list = [
        food_id
        for food_id in cache.restaurant_data
        if isinstance(food_id, int) and len(cache.restaurant_data[food_id])
    ]
    now_food_id = random.choice(food_list)
    now_food = cache.restaurant_data[now_food_id][
        random.choice(list(cache.restaurant_data[now_food_id].keys()))
    ]
    character_data.food_bag[now_food.uid] = now_food
    del cache.restaurant_data[now_food_id][now_food.uid]


@handle_state_machine.add_state_machine(constant.StateMachine.MOVE_TO_RAND_RESTAURANT)
def character_move_to_rand_restaurant(character_id: int):
    """
    设置角色状态为向随机就餐区移动
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache.character_data[character_id]
    to_restaurant = map_handle.get_map_system_path_for_str(random.choice(cache.place_data["Restaurant"]))
    _, _, move_path, move_time = character_move.character_move(
        character_id,
        map_handle.get_map_system_path_for_str(character_data.classroom),
    )
    character_data.behavior["BehaviorId"] = constant.Behavior.MOVE
    character_data.behavior["MoveTarget"] = move_path
    character_data.behavior["Duration"] = move_time
    character_data.state = constant.CharacterStatus.STATUS_MOVE


@handle_state_machine.add_state_machine(constant.StateMachine.EAT_BAG_RAND_FOOD)
def character_eat_rand_food(character_id: int):
    """
    角色随机食用背包中的食物
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache.character_data[character_id]
    character_data.behavior["BehaviorId"] = constant.Behavior.EAT
    character_data.behavior["EatFood"] = character_data.food_bag[
        random.choice(list(character_data.food_bag.keys()))
    ]
    character_data.behavior["Duration"] = 1
    character_data.state = constant.CharacterStatus.STATUS_EAT
