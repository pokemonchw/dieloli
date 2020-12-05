from functools import wraps
from typing import Dict
from types import FunctionType
from Script.Core import cache_control, game_type, constant
from Script.Design import character

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


def add_target(target: str, premise_list: set, effect_list: set) -> FunctionType:
    """
    添加目标
    Keyword arguments:
    target -- 目标id
    premise_list - 目标前提集合
    effect_list -- 目标效果集合
    Return arguments:
    FunctionType -- 目标执行函数对象
    """

    def decoraror(func):
        @wraps(func)
        def return_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        cache.handle_target_data[target] = return_wrapper
        cache.premise_target_table[target] = premise_list
        for effect in effect_list:
            cache.effect_target_table.setdefault(effect, set())
            cache.effect_target_table[effect].add(target)
        return return_wrapper

    return decoraror


@add_target(
    "EatBreakFastInRestaurant",
    {
        constant.Premise.IN_RESTAURANT,
        constant.Premise.IN_BREAKFAST_TIME,
        constant.Premise.HUNGER,
        constant.Premise.HAVE_FOOD,
    },
    {},
)
def handle_eat_break_fast_in_restaurant(character_id: int):
    """
    控制角色在餐厅吃早餐
    Keyword arguments:
    character_id -- 角色id
    """
    character.character_eat_rand_food(character_id)


@add_target(
    "EatLunchInRestaurant",
    {
        constant.Premise.IN_RESTAURANT,
        constant.Premise.IN_LUNCH_TIME,
        constant.Premise.HUNGER,
        constant.Premise.HAVE_FOOD,
    },
    {},
)
def handle_eat_lunch_in_restaruant(character_id: int):
    """
    控制角色在餐厅吃午餐
    Keyword arguments:
    character_id -- 角色id
    """
    character.character_eat_rand_food(character_id)


@add_target(
    "EatDinnerInRestaurant",
    {
        constant.Premise.IN_RESTAURANT,
        constant.Premise.IN_DINNER_TIME,
        constant.Premise.HUNGER,
        constant.Premise.HAVE_FOOD,
    },
    {},
)
def handle_eat_dinner_in_restaurant(character_id: int):
    """
    控制角色在餐厅吃晚餐
    """
    character.character_eat_rand_food(character_id)


@add_target(
    "GoCafeteria",
    {constant.Premise.HUNGER, constant.Premise.NOT_HAVE_FOOD},
    {constant.Premise.IN_CAFETERIA},
)
def handle_go_cafeteria(character_id: int):
    """
    控制角色前往随机取餐区
    Keyword arguments:
    character_id -- 角色id
    """
    character.character_move_to_rand_cafeteria(character_id)


@add_target(
    "GoRestaurant",
    {constant.Premise.HUNGER, constant.Premise.HAVE_FOOD},
    {constant.Premise.IN_RESTAURANT},
)
def handle_go_restaturant(character_id: int):
    """
    控制角色前往随机就餐区
    Keyword arguments:
    character_id -- 角色id
    """
    character.character_move_to_rand_restaurant(character_id)


@add_target(
    "BuyFood",
    {
        constant.Premise.HUNGER,
        constant.Premise.IN_CAFETERIA,
        constant.Premise.NOT_HAVE_FOOD,
    },
    {constant.Premise.HAVE_FOOD},
)
def handle_buy_food(character_id: int):
    """
    控制角色购买食物
    Keyword arguments:
    character_id -- 角色id
    """
    character.character_buy_rand_food_at_restaurant(character_id)
