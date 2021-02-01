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
    character_data: game_type.Character = cache.character_data[character_id]
    _, _, move_path, move_time = character_move.character_move(
        character_id,
        map_handle.get_map_system_path_for_str(character_data.classroom),
    )
    character_data.behavior.behavior_id = constant.Behavior.MOVE
    character_data.behavior.move_target = move_path
    character_data.behavior.duration = move_time
    character_data.state = constant.CharacterStatus.STATUS_MOVE


@handle_state_machine.add_state_machine(constant.StateMachine.MOVE_TO_RAND_CAFETERIA)
def character_move_to_rand_cafeteria(character_id: int):
    """
    移动至随机取餐区
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    to_cafeteria = map_handle.get_map_system_path_for_str(random.choice(cache.place_data["Cafeteria"]))
    _, _, move_path, move_time = character_move.character_move(character_id, to_cafeteria)
    character_data.behavior.behavior_id = constant.Behavior.MOVE
    character_data.behavior.move_target = move_path
    character_data.behavior.duration = move_time
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
    character_data: game_type.Character = cache.character_data[character_id]
    to_restaurant = map_handle.get_map_system_path_for_str(random.choice(cache.place_data["Restaurant"]))
    _, _, move_path, move_time = character_move.character_move(
        character_id,
        map_handle.get_map_system_path_for_str(character_data.classroom),
    )
    character_data.behavior.behavior_id = constant.Behavior.MOVE
    character_data.behavior.move_target = move_path
    character_data.behavior.duration = move_time
    character_data.state = constant.CharacterStatus.STATUS_MOVE


@handle_state_machine.add_state_machine(constant.StateMachine.EAT_BAG_RAND_FOOD)
def character_eat_rand_food(character_id: int):
    """
    角色随机食用背包中的食物
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.behavior.behavior_id = constant.Behavior.EAT
    character_data.behavior.eat_food = character_data.food_bag[
        random.choice(list(character_data.food_bag.keys()))
    ]
    character_data.behavior.duration = 1
    character_data.state = constant.CharacterStatus.STATUS_EAT


@handle_state_machine.add_state_machine(constant.StateMachine.CHAT_RAND_CHARACTER)
def character_chat_rand_character(character_id: int):
    """
    角色和场景内随机角色聊天
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_list = list(
        cache.scene_data[
            map_handle.get_map_system_path_str_for_list(character_data.position)
        ].character_list
    )
    character_list.remove(character_id)
    target_id = random.choice(character_list)
    character_data.behavior.behavior_id = constant.Behavior.CHAT
    character_data.behavior.duration = 10
    character_data.target_character_id = target_id
    character_data.state = constant.CharacterStatus.STATUS_CHAT


@handle_state_machine.add_state_machine(constant.StateMachine.WEAR_CLEAN_UNDERWEAR)
def character_wear_clean_underwear(character_id: int):
    """
    角色穿着干净的上衣
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 1 in character_data.clothing:
        value_dict = {}
        for clothing in character_data.clothing[1]:
            clothing_data: game_type.Clothing = character_data.clothing[1][clothing]
            value_dict[clothing_data.cleanliness] = clothing
        now_value = max(value_dict.keys())
        character_data.put_on[1] = value_dict[now_value]


@handle_state_machine.add_state_machine(constant.StateMachine.WEAR_CLEAN_UNDERPANTS)
def character_wear_clean_underpants(character_id: int):
    """
    角色穿着干净的内裤
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 7 in character_data.clothing:
        value_dict = {}
        for clothing in character_data.clothing[7]:
            clothing_data: game_type.Clothing = character_data.clothing[7][clothing]
            value_dict[clothing_data.cleanliness] = clothing
        now_value = max(value_dict.keys())
        character_data.put_on[7] = value_dict[now_value]


@handle_state_machine.add_state_machine(constant.StateMachine.WEAR_CLEAN_BRA)
def character_wear_clean_bra(character_id: int):
    """
    角色穿着干净的胸罩
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 6 in character_data.clothing:
        value_dict = {}
        for clothing in character_data.clothing[6]:
            clothing_data: game_type.Clothing = character_data.clothing[6][clothing]
            value_dict[clothing_data.cleanliness] = clothing
        now_value = max(value_dict.keys())
        character_data.put_on[6] = value_dict[now_value]


@handle_state_machine.add_state_machine(constant.StateMachine.WEAR_CLEAN_PANTS)
def character_wear_clean_pants(character_id: int):
    """
    角色穿着干净的裤子
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 2 in character_data.clothing:
        value_dict = {}
        for clothing in character_data.clothing[2]:
            clothing_data: game_type.Clothing = character_data.clothing[2][clothing]
            value_dict[clothing_data.cleanliness] = clothing
        now_value = max(value_dict.keys())
        character_data.put_on[2] = value_dict[now_value]


@handle_state_machine.add_state_machine(constant.StateMachine.WEAR_CLEAN_SKIRT)
def character_wear_clean_skirt(character_id: int):
    """
    角色穿着干净的短裙
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 3 in character_data.clothing:
        value_dict = {}
        for clothing in character_data.clothing[3]:
            clothing_data: game_type.Clothing = character_data.clothing[3][clothing]
            value_dict[clothing_data.cleanliness] = clothing
        now_value = max(value_dict.keys())
        character_data.put_on[3] = value_dict[now_value]


@handle_state_machine.add_state_machine(constant.StateMachine.WEAR_CLEAN_SHOES)
def character_wear_clean_shoes(character_id: int):
    """
    角色穿着干净的鞋子
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 4 in character_data.clothing:
        value_dict = {}
        for clothing in character_data.clothing[4]:
            clothing_data: game_type.Clothing = character_data.clothing[4][clothing]
            value_dict[clothing_data.cleanliness] = clothing
        now_value = max(value_dict.keys())
        character_data.put_on[4] = value_dict[now_value]


@handle_state_machine.add_state_machine(constant.StateMachine.WEAR_CLEAN_SOCKS)
def character_wear_clean_socks(character_id: int):
    """
    角色穿着干净的袜子
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 5 in character_data.clothing:
        value_dict = {}
        for clothing in character_data.clothing[5]:
            clothing_data: game_type.Clothing = character_data.clothing[5][clothing]
            value_dict[clothing_data.cleanliness] = clothing
        now_value = max(value_dict.keys())
        character_data.put_on[5] = value_dict[now_value]
