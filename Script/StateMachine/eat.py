import random
from Script.Design import handle_state_machine, constant
from Script.Core import cache_control, game_type
from Script.Config import game_config

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_state_machine.add_state_machine(constant.StateMachine.EAT_BAG_RAND_FOOD)
def character_eat_rand_food(character_id: int):
    """
    角色随机食用背包中的食物
    Keyword arguments:
    character_id -- 角色id
    """
    import uuid
    from Script.Design import session_handler, map_handle
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
    
    # Check if in cafeteria (Map 10 or 16)
    in_cafeteria = False
    if character_data.position and len(character_data.position) > 0:
        in_cafeteria = character_data.position[0] in [10, 16]
        
    if in_cafeteria:
        character_data.behavior.duration = 10
        # Start DiningSession
        scene_path_str = map_handle.get_map_system_path_str_for_list(character_data.position)
        scene_data: game_type.Scene = cache.scene_data[scene_path_str]
        session_uid = str(uuid.uuid4())
        session = game_type.InteractionSession(character_id, [], constant.Behavior.EAT)
        session.uid = session_uid
        session.start_time = cache.game_time
        session.is_pending = False
        cache.interaction_sessions[session_uid] = session
        scene_data.social_fields[session_uid] = "Dining"
        
        handler = session_handler.get_session_handler(session_uid)
        if handler:
            handler.on_start()
            
        character_data.state = constant.CharacterStatus.STATUS_SOCIAL_INTERACTING
        character_data.active_session = session_uid
    else:
        character_data.behavior.duration = 1
        character_data.state = constant.CharacterStatus.STATUS_EAT
        
    character_data.behavior.food_quality = character_data.behavior.eat_food.quality
    food_name = ""
    if character_data.behavior.eat_food.recipe != -1:
        food_recipe: game_type.Recipes = cache.recipe_data[character_data.behavior.eat_food.recipe]
        food_name = food_recipe.name
    else:
        food_config = game_config.config_food[character_data.behavior.eat_food.id]
        food_name = food_config.name
    character_data.behavior.food_name = food_name


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
    character_data.behavior.food_quality = character_data.behavior.eat_food.quality
    food_name = ""
    if character_data.behavior.eat_food.recipe != -1:
        food_recipe: game_type.Recipes = cache.recipe_data[character_data.behavior.eat_food.recipe]
        food_name = food_recipe.name
    else:
        food_config = game_config.config_food[character_data.behavior.eat_food.id]
        food_name = food_config.name
    character_data.behavior.food_name = food_name
    character_data.state = constant.CharacterStatus.STATUS_EAT

