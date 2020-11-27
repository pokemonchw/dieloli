import random
import uuid
import datetime
from Script.Core import (
    cache_contorl,
    value_handle,
    constant,
    game_type,
)
from Script.Design import (
    attr_calculation,
    clothing,
    nature,
    map_handle,
    character_move,
)
from Script.Config import game_config


cache:game_type.Cache = cache_contorl.cache
""" 游戏缓存数据 """


def init_attr(character_id: int):
    """
    初始化角色属性
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache.character_data[character_id]
    character_data.language[character_data.mother_tongue] = 10000
    character_data.birthday = attr_calculation.get_rand_npc_birthday(character_data.age)
    character_data.end_age = attr_calculation.get_end_age(character_data.sex)
    character_data.height = attr_calculation.get_height(character_data.sex, character_data.age)
    bmi = attr_calculation.get_bmi(character_data.weigt_tem)
    character_data.weight = attr_calculation.get_weight(bmi, character_data.height.now_height)
    character_data.bodyfat = attr_calculation.get_body_fat(character_data.sex, character_data.bodyfat_tem)
    character_data.measurements = attr_calculation.get_measurements(
        character_data.sex,
        character_data.height.now_height,
        character_data.weight,
        character_data.bodyfat,
        character_data.bodyfat_tem,
    )
    character_data.sex_experience = attr_calculation.get_sex_experience(
        character_data.sex_experience_tem, character_data.sex
    )
    default_clothing_data = clothing.creator_suit(character_data.clothing_tem, character_data.sex)
    character_data.clothing = {
        clothing: {default_clothing_data[clothing].uid: default_clothing_data[clothing]}
        for clothing in default_clothing_data
    }
    character_data.chest = attr_calculation.get_chest(character_data.chest_tem, character_data.birthday)
    character_data.hit_point_max = attr_calculation.get_max_hit_point(character_data.hit_point_tem)
    character_data.hit_point = character_data.hit_point_max
    character_data.mana_point_max = attr_calculation.get_max_mana_point(character_data.mana_point_tem)
    character_data.mana_point = character_data.mana_point_max
    new_nature = nature.get_random_nature()
    for nature_id in new_nature:
        if nature_id not in character_data.nature:
            character_data.nature[nature_id] = new_nature[nature_id]
    init_class(character_data)


def init_class(character_data: game_type.Character):
    """
    初始化角色班级
    character_data -- 角色对象
    """
    if character_data.age <= 18 and character_data.age >= 7:
        class_grade = str(character_data.age - 6)
        character_data.classroom = random.choice(cache.place_data["Classroom_" + class_grade])


def init_character_behavior_start_time(character_id: int):
    """
    将角色的行动开始时间同步为当前游戏时间
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache.character_data[character_id]
    game_time = cache.game_time
    start_time = datetime.datetime(
        game_time.year,
        game_time.month,
        game_time.day,
        game_time.hour,
        game_time.minute,
    )
    character_data.behavior.start_time = start_time


def character_move_to_classroom(character_id: int):
    """
    设置角色行为状态为向所属教室移动
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


def character_attend_class(character_id: int):
    """
    设置角色行为状态为上课
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache.character_data[character_id]
    character_data.behavior["BehaviorId"] = constant.Behavior.ATTEND_CLASS
    character_data.behavior["Duration"] = now_time_slice["EndCourse"]
    character_data.behavior["MoveTarget"] = []
    character_data.state = constant.CharacterStatus.STATUS_ATTEND_CLASS
    init_character_behavior_start_time(character_id)


def character_move_to_rand_cafeteria(character_id: int):
    """
    设置角色状态为向随机取餐区移动
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache.character_data[character_id]
    to_cafeteria = map_handle.get_map_system_path_for_str(
        random.choice(cache.place_data["Cafeteria"])
    )
    _, _, move_path, move_time = character_move.character_move(character_id, to_cafeteria)
    character_data.behavior["BehaviorId"] = constant.Behavior.MOVE
    character_data.behavior["MoveTarget"] = move_path
    character_data.behavior["Duration"] = move_time
    character_data.state = constant.CharacterStatus.STATUS_MOVE


def character_move_to_rand_restaurant(character_id: int):
    """
    设置角色状态为向随机就餐区移动
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache.character_data[character_id]
    to_restaurant = map_handle.get_map_system_path_for_str(
        random.choice(cache.place_data["Restaurant"])
    )
    _, _, move_path, move_time = character_move.character_move(
        character_id,
        map_handle.get_map_system_path_for_str(character_data.classroom),
    )
    character_data.behavior["BehaviorId"] = constant.Behavior.MOVE
    character_data.behavior["MoveTarget"] = move_path
    character_data.behavior["Duration"] = move_time
    character_data.state = constant.CharacterStatus.STATUS_MOVE


def character_rest_to_time(character_id: int, need_time: int):
    """
    设置角色状态为休息指定时间
    Keyword arguments:
    character_id -- 角色id
    need_time -- 休息时长(分钟)
    """
    character_data = cache.character_data[character_id]
    character_data.behavior["Duration"] = need_time
    character_data.behavior["BehaviorId"] = constant.Behavior.REST
    character_data.state = constant.CharacterStatus.STATUS_REST


def character_buy_rand_food_at_restaurant(character_id: int):
    """
    角色在取餐区中随机获取一种食物放入背包
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
