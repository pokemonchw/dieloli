from functools import wraps
from typing import Dict
from Script.Core import cache_contorl, game_type
from Script.Design import character


def add_target(target: str, premise_list: set, effect_list: set) -> callable:
    """
    添加目标
    Keyword arguments:
    target -- 目标id
    premise_list - 目标前提集合
    effect_list -- 目标效果集合
    Return arguments:
    callable -- 目标执行函数对象
    """

    def decoraror(func):
        @wraps(func)
        def return_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        cache_contorl.handle_target_data[target] = return_wrapper
        cache_contorl.premise_target_table[target] = premise_list
        for effect in effect_list:
            cache_contorl.effect_target_table.setdefault(effect, set())
            cache_contorl.effect_target_table[effect].add(target)
        return return_wrapper

    return decoraror


@add_target(
    "EatBreakFastInRestaurant", {"InRestaurant", "InBreakfastTime", "Hunger", "HaveFood"}, {},
)
def handle_eat_break_fast_in_restaurant(character_id: int):
    """
    控制角色在餐厅吃早餐
    Keyword arguments:
    character_id -- 角色id
    """
    character.character_eat_rand_food(character_id)


@add_target(
    "EatLunchInRestaurant", {"InRestaurant", "InLunchTime", "Hunger", "HaveFood"}, {},
)
def handle_eat_lunch_in_restaruant(character_id: int):
    """
    控制角色在餐厅吃午餐
    Keyword arguments:
    character_id -- 角色id
    """
    character.character_eat_rand_food(character_id)


@add_target(
    "EatDinnerInRestaurant", {"InRestaurant", "InDinnerTime", "Hunger", "HaveFood"}, {},
)
def handle_eat_dinner_in_restaurant(character_id: int):
    """
    控制角色在餐厅吃晚餐
    """
    character.character_eat_rand_food(character_id)


@add_target(
    "GoCafeteria", {"Hunger", "NotHaveFood"}, {"InCafeteria"},
)
def handle_go_cafeteria(character_id: int):
    """
    控制角色前往随机取餐区
    Keyword arguments:
    character_id -- 角色id
    """
    character.character_move_to_rand_cafeteria(character_id)


@add_target(
    "GoRestaurant", {"Hunger", "HaveFood"}, {"InRestaurant"},
)
def handle_go_restaturant(character_id: int):
    """
    控制角色前往随机就餐区
    Keyword arguments:
    character_id -- 角色id
    """
    character.character_move_to_rand_restaurant(character_id)


@add_target(
    "BuyFood", {"Hunger", "InCafeteria", "NotHaveFood"}, {"HaveFood"},
)
def handle_buy_food(character_id: int):
    """
    控制角色购买食物
    Keyword arguments:
    character_id -- 角色id
    """
    character.character_buy_rand_food_at_restaurant(character_id)
