import math
from functools import wraps
from Script.Core import cache_contorl, constant
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
    character_data = cache_contorl.character_data[character_id]
    if character_data.course.course_index <= 2 and not character_data.course.in_course:
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


@add_premise("HaveTarget")
def handle_have_target(character_id:int) -> int:
    """
    校验角色是否有交互对象
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    if character_data.target_character_id == character_id:
        return 0
    return 1


@add_premise("HaveItemByTagDraw")
def handle_have_item_by_tag_draw(character_id:int) -> int:
    """
    校验角色是否拥有绘画类道具
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    for item in character_data.item:
        if character_data.item[item]["ItemTag"] == "Draw":
            return 1
    return 0


@add_premise("HaveItemByTagShooting")
def handle_have_item_by_tag_shooting(character_id:int) -> int:
    """
    校验角色是否拥有射击类道具
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    for item in character_data.item:
        if character_data.item[item]["ItemTag"] == "Shooting":
            return 1
    return 0


@add_premise("HaveGuitar")
def handle_have_guitar(character_id:int) -> int:
    """
    校验角色是否拥有吉他
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    for item in character_data.item:
        if item == "Guitar":
            return 1
    return 0


@add_premise("HaveHarmonica")
def handle_have_harmonica(character_id:int) -> int:
    """
    校验角色是否拥有口琴
    Keyword arguments:
    character_id --角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    for item in character_data.item:
        if item == "Harmonica":
            return 1
    return 0


@add_premise("HaveBamBooFlute")
def handle_have_bamboogflute(character_id:int) -> int:
    """
    校验角色是否拥有竹笛
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    for item in character_data.item:
        if item == "BamBooFlute":
            return 1
    return 0


@add_premise("HaveBasketBall")
def handle_have_basketball(character_id:int) -> int:
    """
    校验角色是否拥有篮球
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    for item in character_data.item:
        if item == "BasketBall":
            return 1
    return 0


@add_premise("HaveFootBall")
def handle_have_football(character_id:int) -> int:
    """
    校验角色是否拥有足球
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    for item in character_data.item:
        if item == "FootBall":
            return 1
    return 0


@add_premise("HaveTableTennis")
def handle_have_tabletennis(character_id:int) -> int:
    """
    校验角色是否拥有乒乓球
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    for item in character_data.item:
        if item == "TableTennis":
            return 1
    return 0


@add_premise("InSwimmingPool")
def handle_in_swimming_pool(character_id:int) -> int:
    """
    校验角色是否在游泳池中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    now_position = character_data.position
    now_scene_str = map_handle.get_map_system_path_str_for_list(now_position)
    now_scene_data = cache_contorl.scene_data[now_scene_str]
    if now_scene_data["SceneTag"] == "SwimmingPool":
        return 1
    return 0


@add_premise("InClassroom")
def handle_in_classroom(character_id:int) -> int:
    """
    校验角色是否处于教室中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    now_position = character_data.position
    now_scene_str = map_handle.get_map_system_path_str_for_list(now_position)
    now_scene_data = cache_contorl.scene_data[now_scene_str]
    if now_scene_data["SceneTag"] == "Classroom":
        return 1
    return 0


@add_premise("IsStudent")
def handle_is_student(character_id:int) -> int:
    """
    校验角色是否是学生
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    if character_data.age <= 18:
        return 1
    return 0


@add_premise("IsTeacher")
def handle_is_teacher(character_id:int) -> int:
    """
    校验角色是否是老师
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    if character_data.age > 18:
        return 1
    return 0


@add_premise("InShop")
def handle_in_shop(character_id:int) -> int:
    """
    校验角色是否在商店中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    now_position = character_data.position
    now_scene_str = map_handle.get_map_system_path_str_for_list(now_position)
    now_scene_data = cache_contorl.scene_data[now_scene_str]
    if now_scene_data["SceneTag"] == "Shop":
        return 1
    return 0


@add_premise("InSleepTime")
def handle_in_sleep_time(character_id:int) -> int:
    """
    校验角色当前是否处于睡觉时间
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    now_time = character_data.behavior["StartTime"]
    if int(now_time["hour"]) >= 22 or int(now_time["hour"]) <= 4:
        return 1
    return 0


@add_premise("InSiestaTime")
def handle_in_siesta_time(character_id:int) -> int:
    """
    校验角色是否处于午休时间
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    now_time = character_data.behavior["StartTime"]
    if int(now_time["hour"]) >= 12 or int(now_time["hour"]) <= 15:
        return 1
    return 0


@add_premise("TargetIsFutaOrWoman")
def handle_target_is_futa_or_woman(character_id:int) -> int:
    """
    校验角色的目标对象性别是否为女性或扶她
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    target_data = cache_contorl.character_data[character_data.target_character_id]
    if target_data.sex in {"Futa","Woman"}:
        return 1
    return 0


@add_premise("TargetIsFutaOrMan")
def handle_target_is_futa_or_man(character_id:int) -> int:
    """
    校验角色的目标对象性别是否为男性或扶她
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    target_data = cache_contorl.character_data[character_data.target_character_id]
    if target_data.sex in {"Man","Woman"}:
        return 1
    return 0


@add_premise("TargetNotPutUnderwear")
def handle_target_not_put_underwear(character_id:int) -> int:
    """
    校验角色的目标对象是否没穿上衣
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    target_data = cache_contorl.character_data[character_data.target_character_id]
    if target_data.put_on["Underwear"] == "":
        return 1
    return 0


@add_premise("TargetPutOnSkirt")
def handle_target_put_on_skirt(character_id:int) -> int:
    """
    校验角色的目标对象是否穿着短裙
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    target_data = cache_contorl.character_data[character_data.target_character_id]
    if target_data.put_on["Skirt"] == "":
        return 0
    return 1
