import random
import uuid
from Script.Core import (
    cache_contorl,
    text_loading,
    value_handle,
    constant,
    game_type,
)
from Script.Design import attr_calculation, clothing, nature


def init_attr(character_data: game_type.Character):
    """
    初始化角色属性
    Keyword arguments:
    character_data -- 角色对象
    """
    character_data.language[character_data.mother_tongue] = 10000
    character_data.birthday = attr_calculation.get_rand_npc_birthday(
        character_data.age
    )
    character_data.end_age = attr_calculation.get_end_age(character_data.sex)
    character_data.height = attr_calculation.get_height(
        character_data.sex, character_data.age
    )
    bmi = attr_calculation.get_bmi(character_data.weigt_tem)
    character_data.weight = attr_calculation.get_weight(
        bmi, character_data.height["NowHeight"]
    )
    character_data.bodyfat = attr_calculation.get_bodyfat(
        character_data.sex, character_data.bodyfat_tem
    )
    character_data.measurements = attr_calculation.get_measurements(
        character_data.sex,
        character_data.height["NowHeight"],
        character_data.weight,
        character_data.bodyfat,
        character_data.bodyfat_tem,
    )
    character_data.sex_experience = attr_calculation.get_sex_experience(
        character_data.sex_experience_tem
    )
    character_data.sex_grade = attr_calculation.get_sex_grade(
        character_data.sex_experience
    )
    default_clothing_data = clothing.creator_suit(
        character_data.clothing_tem, character_data.sex
    )
    character_data.clothing = {
        clothing: {uuid.uuid1(): default_clothing_data[clothing]}
        if clothing in default_clothing_data
        else {}
        for clothing in character_data.clothing
    }
    character_data.chest = attr_calculation.get_chest(
        character_data.chest_tem, character_data.birthday
    )
    character_data.hit_point_max = attr_calculation.get_max_hit_point(
        character_data.hit_point_tem
    )
    character_data.hit_point = character_data.hit_point_max
    character_data.mana_point_max = attr_calculation.get_max_mana_point(
        character_data.mana_point_tem
    )
    character_data.mana_point = character_data.mana_point_max
    character_data.nature = nature.get_random_nature()
    character_data.status = text_loading.get_game_data(
        constant.FilePath.CHARACTER_STATE_PATH
    )
    character_data.wear_item = {
        "Wear": {
            key: {}
            for key in text_loading.get_game_data(
                constant.FilePath.WEAR_ITEM_PATH
            )["Wear"]
        },
        "Item": {},
    }
    character_data.engraving = {
        "Pain": 0,
        "Happy": 0,
        "Yield": 0,
        "Fear": 0,
        "Resistance": 0,
    }
    character_data.social_contact = {
        social: {}
        for social in text_loading.get_text_data(
            constant.FilePath.STAGE_WORD_PATH, "144"
        )
    }
    init_class(character_data)
    put_on_clothing(character_data)
    if character_data.occupation == "":
        if character_data.age <= 18:
            character_data.occupation = "Student"
        else:
            character_data.occupation = "Teacher"


def init_class(character_data: game_type.Character):
    """
    初始化角色班级
    character_data -- 角色对象
    """
    if character_data.age <= 18 and character_data.age >= 7:
        class_grade = str(character_data.age - 6)
        character_data.classroom = random.choice(
            cache_contorl.place_data["Classroom_" + class_grade]
        )


def put_on_clothing(character_data: game_type.Character):
    """
    角色自动选择并穿戴服装
    Keyword arguments:
    character_data -- 角色对象
    """
    character_clothing_data = character_data.clothing
    collocation_data = {}
    clothings_name_data = clothing.get_clothing_name_data(
        character_clothing_data
    )
    clothings_price_data = clothing.get_clothing_price_data(
        character_clothing_data
    )
    for clothing_type in clothings_name_data:
        clothing_type_data = clothings_name_data[clothing_type]
        for clothing_name in clothing_type_data:
            clothing_name_data = clothing_type_data[clothing_name]
            clothing_id = list(clothing_name_data.keys())[-1]
            clothing_data = character_clothing_data[clothing_type][clothing_id]
            now_collocation_data = clothing.get_clothing_collocation_data(
                clothing_data,
                clothing_type,
                clothings_name_data,
                clothings_price_data,
                character_clothing_data,
            )
            if now_collocation_data != "None":
                now_collocation_data[clothing_type] = clothing_id
                now_collocation_data["Price"] += clothings_price_data[
                    clothing_type
                ][clothing_id]
                collocation_data[clothing_id] = now_collocation_data
    collocation_price_data = {
        collocation: collocation_data[collocation]["Price"]
        for collocation in collocation_data
    }
    collocation_id = list(
        value_handle.sorted_dict_for_values(collocation_price_data).keys()
    )[-1]
    character_data.put_on = collocation_data[collocation_id]


def init_character_behavior_start_time(character_id: int):
    """
    将角色的行动开始时间同步为当前游戏时间
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache_contorl.character_data[character_id]
    character_data.behavior["StartTime"]["year"] = cache_contorl.game_time[
        "year"
    ]
    character_data.behavior["StartTime"]["month"] = cache_contorl.game_time[
        "month"
    ]
    character_data.behavior["StartTime"]["day"] = cache_contorl.game_time[
        "day"
    ]
    character_data.behavior["StartTime"]["hour"] = cache_contorl.game_time[
        "hour"
    ]
    character_data.behavior["StartTime"]["minute"] = cache_contorl.game_time[
        "minute"
    ]


def character_move_to_classroom(character_id: int):
    """
    设置角色行为状态为向所属教室移动
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache_contorl.character_data[character_id]
    _, _, move_path, move_time = character_move.character_move(
        character_id,
        map_handle.get_map_system_path_for_str(
            character_data.classroom
        ),
    )
    character_data.behavior["BehaviorId"] = constant.Behavior.MOVE
    character_data.behavior["MoveTarget"] = move_path
    character_data.behavior["Duration"] = move_time
    character_data.state = constant.CharacterStatus.STATUS_MOVE


def character_attend_class(character_id:int):
    """
    设置角色行为状态为上课
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache_contorl.character_data[character_id]
    character_data.behavior[
        "BehaviorId"
    ] = constant.Behavior.ATTEND_CLASS
    character_data.behavior["Duration"] = now_time_slice["EndCourse"]
    character_data.behavior["MoveTarget"] = []
    character_data.state = constant.CharacterStatus.STATUS_ATTEND_CLASS
    init_character_behavior_start_time(character_id)


def character_move_to_rand_cafeteria(character_id:int):
    """
    设置角色状态为向随机取餐区移动
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache_contorl.character_data[character_id]
    to_cafeteria = map_handle.get_map_system_path_for_str(random.choice(cache_contorl.place_data["Cafeteria"]))
    _,_,move_path,move_time = character_move.character_move(character_id,to_cafeteria)
    character_data.behavior["BehaviorId"] = constant.Behavior.MOVE
    character_data.behavior["MoveTarget"] = move_path
    character_data.behavior["Duration"] = move_time
    character_data.state = constant.CharacterStatus.STATUS_MOVE
    init_character_behavior_start_time(character_id)


def character_move_to_rand_restaurant(character_id:int):
    """
    设置角色状态为向随机就餐区移动
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache_contorl.character_data[character_id]
    to_restaurant = map_handle.get_map_system_path_for_str(random.choice(cache_contorl.place_data["Restaurant"]))
    _, _, move_path, move_time = character_move.character_move(
        character_id,
        map_handle.get_map_system_path_for_str(
            character_data.classroom
        ),
    )
    character_data.behavior["BehaviorId"] = constant.Behavior.MOVE
    character_data.behavior["MoveTarget"] = move_path
    character_data.behavior["Duration"] = move_time
    character_data.state = constant.CharacterStatus.STATUS_MOVE
    init_character_behavior_start_time(character_id)


def character_rest_to_time(character_id:int,need_time:int):
    """
    设置角色状态为休息指定时间
    Keyword arguments:
    character_id -- 角色id
    need_time -- 休息时长(分钟)
    """
    character_data = cache_contorl.character_data[character_id]
    character_data.behavior["Duration"] = need_time
    character_data.behavior["BehaviorId"] = constant.Behavior.REST
    character_data.state = constant.CharacterStatus.STATUS_REST
    init_character_behavior_start_time(character_id)


def character_buy_rand_food_at_restaurant(character_id:int):
    """
    角色在取餐区中随机获取一种食物放入背包
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache_contorl.character_data[character_id]
    food_list = [food_id for food_id in cache_contorl.restaurant_data if isinstance(food_id,int) and len(cache_contorl.restaurant_data[food_id])]
    now_food_id = random.choice(food_list)
    now_food = cache_contorl.restaurant_data[now_food_id][random.choice(list(cache_contorl.restaurant_data[now_food_id].keys()))]
    character_data.food_bag[now_food.uid] = now_food
    del cache_contorl.restaurant_data[now_food_id][now_food.uid]


def character_eat_rand_food(character_id:int):
    """
    角色随机食用背包中的食物
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache_contorl.character_data[character_id]
    character_data.behavior["BehaviorId"] = constant.Behavior.EAT
    character_data.behavior["EatFood"] = character_data.food_bag[random.choice(list(character_data.food_bag.keys()))]
    character_data.behavior["Duration"] = 10
    character_data.state = constant.CharacterStatus.STATUS_EAT
    init_character_behavior_start_time(character_id)
