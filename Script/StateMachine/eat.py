import random
from Script.Design import handle_state_machine, constant
from Script.Core import cache_control, game_type

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_state_machine.add_state_machine(constant.StateMachine.EAT_BAG_RAND_FOOD)
def character_eat_rand_food(character_id: int):
    """
    角色随机食用背包中的食物
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.behavior.behavior_id = constant.Behavior.EAT
    now_food_list = []
    for food_id in character_data.food_bag:
        now_food: game_type.Food = character_data.food_bag[food_id]
        if 27 in now_food.feel and now_food.eat:
            now_food_list.append(food_id)
    if not now_food_list:
        return
    character_data.behavior.eat_food = character_data.food_bag[random.choice(now_food_list)]
    character_data.behavior.duration = 1
    character_data.behavior.food_quality = now_food.quality
    food_name = ""
    if now_food.recipe != -1:
        food_recipe: game_type.Recipes = cache.recipe_data[now_food.recipe]
        food_name = food_recipe.name
    else:
        food_config = game_config.config_food[now_food.id]
        food_name = food_config.name
    character_data.behavior.food_name = food_name
    character_data.state = constant.CharacterStatus.STATUS_EAT


@handle_state_machine.add_state_machine(constant.StateMachine.DRINK_RAND_DRINKS)
def character_drink_rand_drinks(character_id: int):
    """
    角色饮用背包内的随机饮料
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.behavior.behavior_id = constant.Behavior.EAT
    drink_list = []
    food_list = []
    for food_id in character_data.food_bag:
        now_food: game_type.Food = character_data.food_bag[food_id]
        if 28 in now_food.feel and now_food.eat:
            if 27 in now_food.feel and now_food.feel[27] > now_food.feel[28]:
                food_list.append(food_id)
            else:
                drink_list.append(food_id)
    if drink_list:
        now_list = drink_list
    else:
        now_list = food_list
    if not now_list:
        return
    character_data.behavior.eat_food = character_data.food_bag[random.choice(now_list)]
    character_data.behavior.duration = 1
    character_data.behavior.food_quality = now_food.quality
    food_name = ""
    if now_food.recipe != -1:
        food_recipe: game_type.Recipes = cache.recipe_data[now_food.recipe]
        food_name = food_recipe.name
    else:
        food_config = game_config.config_food[now_food.id]
        food_name = food_config.name
    character_data.behavior.food_name = food_name
    character_data.state = constant.CharacterStatus.STATUS_EAT

