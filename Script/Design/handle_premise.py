import math
from functools import wraps
from Script.Core import cache_contorl,constant
from Script.Design import map_handle, game_time


def add_premise(premise: str) -> callable:
    """
    添加前提
    Keyword arguments:
    premise -- 前提id
    Return arguments:
    callable -- 前提处理函数对象
    """

    def decoraror(func):
        @wraps(func)
        def return_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        cache_contorl.handle_premise_data[premise] = return_wrapper
        return return_wrapper

    return decoraror


def handle_premise(premise: str, character_id: int) -> int:
    """
    调用前提id对应的前提处理函数
    Keyword arguments:
    premise -- 前提id
    character_id -- 角色id
    Return arguments:
    int -- 前提权重加成
    """
    if premise in handle_premise_data:
        return handle_premise_data[premise](character_id)
    else:
        return 0


@add_premise("InCafeteria")
def handle_in_cafeteria(character_id: int) -> int:
    """
    校验角色是否处于取餐区中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    now_position = character_data.position
    now_scene_str = map_handle.get_map_system_path_str_for_list(now_position)
    now_scene_data = cache_contorl.scene_data[now_scene_str]
    if now_scene_data["SceneTag"] == "Cafeteria":
        return 1
    return 0


@add_premise("InRestaurant")
def handle_in_restaurant(character_id: int) -> int:
    """
    校验角色是否处于就餐区中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    now_position = character_data.position
    now_scene_str = map_handle.get_map_system_path_str_for_list(now_position)
    now_scene_data = cache_contorl.scene_data[now_scene_str]
    if now_scene_data["SceneTag"] == "Restaurant":
        return 1
    return 0


@add_premise("InBreakfastTime")
def handle_in_breakfast_time(character_id: int) -> int:
    """
    校验当前时间是否处于早餐时间段
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    now_time_slice = game_time.get_now_time_slice(character_id)
    if now_time_slice["TimeSlice"] == constant.TimeSlice.TIME_BREAKFAST:
        return 1
    return 0


@add_premise("Hunger")
def handle_hunger(character_id: int) -> int:
    """
    校验角色是否处于饥饿状态
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    return math.floor(character_data.status["BodyFeeling"]["Hunger"] / 10)


@add_premise("HaveFood")
def handle_have_food(character_id: int) -> int:
    """
    校验角色是否拥有食物
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    food_index = 0
    for food_id in character_data.food_bag:
        if character_data.food_bag[food_id].eat:
            food_index += 1
    return food_index


@add_premise("NotHaveFood")
def handle_not_have_food(character_id: int) -> int:
    """
    校验角色是否没有食物
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    food_index = 1
    for food_id in character_data.food_bag:
        if character_data.food_bag[food_id].eat:
            return 0
    return food_index
