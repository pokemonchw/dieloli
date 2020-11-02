import math
import datetime
from functools import wraps
from types import FunctionType
from Script.Core import cache_contorl, constant
from Script.Design import map_handle, game_time, attr_calculation


def add_premise(premise: int) -> FunctionType:
    """
    添加前提
    Keyword arguments:
    premise -- 前提id
    Return arguments:
    FunctionType -- 前提处理函数对象
    """

    def decoraror(func):
        @wraps(func)
        def return_wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        cache_contorl.handle_premise_data[premise] = return_wrapper
        return return_wrapper

    return decoraror


def handle_premise(premise: int, character_id: int) -> int:
    """
    调用前提id对应的前提处理函数
    Keyword arguments:
    premise -- 前提id
    character_id -- 角色id
    Return arguments:
    int -- 前提权重加成
    """
    if premise in cache_contorl.handle_premise_data:
        return cache_contorl.handle_premise_data[premise](character_id)
    else:
        return 0


@add_premise(constant.Premise.IN_CAFETERIA)
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


@add_premise(constant.Premise.IN_RESTAURANT)
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


@add_premise(constant.Premise.IN_BREAKFAST_TIME)
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


@add_premise(constant.Premise.IN_LUNCH_TIME)
def handle_in_lunch_time(character_id: int) -> int:
    """
    校验当前是否处于午餐时间段
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    if (
        character_data.course.course_index >= 4
        and character_data.course.course_index <= 6
    ):
        return 1
    return 0


@add_premise(constant.Premise.IN_DINNER_TIME)
def handle_in_dinner_time(character_id: int) -> int:
    """
    校验当前是否处于晚餐时间段
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    if character_data.course.course_index >= 7:
        return 1
    return 0


@add_premise(constant.Premise.HUNGER)
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


@add_premise(constant.Premise.HAVE_FOOD)
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


@add_premise(constant.Premise.NOT_HAVE_FOOD)
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


@add_premise(constant.Premise.HAVE_TARGET)
def handle_have_target(character_id: int) -> int:
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


@add_premise(constant.Premise.TARGET_NO_PLAYER)
def handle_target_no_player(character_id: int) -> int:
    """
    校验角色目标对像是否不是玩家
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    if character_data.target_character_id > 0:
        return 1
    return 0


@add_premise(constant.Premise.HAVE_DRAW_ITEM)
def handle_have_item_by_tag_draw(character_id: int) -> int:
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


@add_premise(constant.Premise.HAVE_SHOOTING_ITEM)
def handle_have_item_by_tag_shooting(character_id: int) -> int:
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


@add_premise(constant.Premise.HAVE_GUITAR)
def handle_have_guitar(character_id: int) -> int:
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


@add_premise(constant.Premise.HAVE_HARMONICA)
def handle_have_harmonica(character_id: int) -> int:
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


@add_premise(constant.Premise.HAVE_BAM_BOO_FLUTE)
def handle_have_bamboogflute(character_id: int) -> int:
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


@add_premise(constant.Premise.HAVE_BASKETBALL)
def handle_have_basketball(character_id: int) -> int:
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


@add_premise(constant.Premise.HAVE_FOOTBALL)
def handle_have_football(character_id: int) -> int:
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


@add_premise(constant.Premise.HAVE_TABLE_TENNIS)
def handle_have_tabletennis(character_id: int) -> int:
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


@add_premise(constant.Premise.IN_SWIMMING_POOL)
def handle_in_swimming_pool(character_id: int) -> int:
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


@add_premise(constant.Premise.IN_CLASSROOM)
def handle_in_classroom(character_id: int) -> int:
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


@add_premise(constant.Premise.IS_STUDENT)
def handle_is_student(character_id: int) -> int:
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


@add_premise(constant.Premise.IS_TEACHER)
def handle_is_teacher(character_id: int) -> int:
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


@add_premise(constant.Premise.IN_SHOP)
def handle_in_shop(character_id: int) -> int:
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


@add_premise(constant.Premise.IN_SLEEP_TIME)
def handle_in_sleep_time(character_id: int) -> int:
    """
    校验角色当前是否处于睡觉时间
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    now_time: datetime.datetime = character_data.behavior["StartTime"]
    if now_time.hour >= 22 or now_time.hour <= 4:
        return 1
    return 0


@add_premise(constant.Premise.IN_SIESTA_TIME)
def handle_in_siesta_time(character_id: int) -> int:
    """
    校验角色是否处于午休时间
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    now_time: datetime.datetime = character_data.behavior["StartTime"]
    if now_time.hour >= 12 or now_time.hour <= 15:
        return 1
    return 0


@add_premise(constant.Premise.TARGET_IS_FUTA_OR_WOMAN)
def handle_target_is_futa_or_woman(character_id: int) -> int:
    """
    校验角色的目标对象性别是否为女性或扶她
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    target_data = cache_contorl.character_data[character_data.target_character_id]
    if target_data.sex in {"Futa", "Woman"}:
        return 1
    return 0


@add_premise(constant.Premise.TARGET_IS_FUTA_OR_MAN)
def handle_target_is_futa_or_man(character_id: int) -> int:
    """
    校验角色的目标对象性别是否为男性或扶她
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    target_data = cache_contorl.character_data[character_data.target_character_id]
    if target_data.sex in {"Man", "Woman"}:
        return 1
    return 0


@add_premise(constant.Premise.IS_MAN)
def handle_is_man(character_id: int) -> int:
    """
    校验角色是否是男性
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    if character_data.sex == "Man":
        return 1
    return 0


@add_premise(constant.Premise.IS_WOMAN)
def handle_is_woman(character_id: int) -> int:
    """
    校验角色是否是女性
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    if character_data.sex == "Woman":
        return 1
    return 0


@add_premise(constant.Premise.TARGET_SAME_SEX)
def handle_target_same_sex(character_id: int) -> int:
    """
    校验角色目标对像是否与自己性别相同
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    target_data = cache_contorl.character_data[character_data.target_character_id]
    if target_data.sex == character_data.sex:
        return 1
    return 0


@add_premise(constant.Premise.TARGET_AGE_SIMILAR)
def handle_target_age_similar(character_id: int) -> int:
    """
    校验角色目标对像是否与自己年龄相差不大
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    target_data = cache_contorl.character_data[character_data.target_character_id]
    if (
        character_data.age >= target_data.age - 2
        and character_data.age <= target_data.age + 2
    ):
        return 1
    return 0


@add_premise(constant.Premise.TARGET_AVERAGE_HEIGHT_SIMILAR)
def handle_target_average_height_similar(character_id: int) -> int:
    """
    校验角色目标身高是否与平均身高相差不大
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    target_data = cache_contorl.character_data[character_data.target_character_id]
    age_tem = attr_calculation.judge_age_group(target_data.age)
    average_height = cache_contorl.average_height_by_age[age_tem][target_data.sex]
    if (
        target_data.height["NowHeight"] >= average_height * 0.95
        and target_data.height["NowHeight"] <= average_height * 1.05
    ):
        return 1
    return 0


@add_premise(constant.Premise.TARGET_AVERAGE_HEIGHT_LOW)
def handle_target_average_height_low(character_id: int) -> int:
    """
    校验角色目标的身高是否低于平均身高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    target_data = cache_contorl.character_data[character_data.target_character_id]
    age_tem = attr_calculation.judge_age_group(target_data.age)
    average_height = cache_contorl.average_height_by_age[age_tem][target_data.sex]
    if target_data.height["NowHeight"] <= average_height * 0.95:
        return 1
    return 0


@add_premise(constant.Premise.TARGET_IS_PLAYER)
def handle_is_player(character_id: int) -> int:
    """
    校验角色目标是否是玩家
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    if character_data.target_character_id == 0:
        return 1
    return 0


@add_premise(constant.Premise.TARGET_AVERGAE_STATURE_SIMILAR)
def handle_target_average_stature_similar(character_id: int) -> int:
    """
    校验角色目体型高是否与平均体型相差不大
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    target_data = cache_contorl.character_data[character_data.target_character_id]
    age_tem = attr_calculation.judge_age_group(target_data.age)
    average_bodyfat = cache_contorl.average_bodyfat_by_age[age_tem][target_data.sex]
    if (
        target_data.bodyfat >= average_bodyfat * 0.95
        and target_data.bodyfat <= average_bodyfat * 1.05
    ):
        return 1
    return 0


@add_premise(constant.Premise.TARGET_NOT_PUT_ON_UNDERWEAR)
def handle_target_not_put_underwear(character_id: int) -> int:
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


@add_premise(constant.Premise.TARGET_NOT_PUT_ON_SKIRT)
def handle_target_put_on_skirt(character_id: int) -> int:
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


@add_premise(constant.Premise.IS_PLAYER)
def handle_is_player(character_id: int) -> int:
    """
    校验是否是玩家角色
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    if not character_id:
        return 1
    return 0


@add_premise(constant.Premise.NO_PLAYER)
def handle_no_player(character_id: int) -> int:
    """
    校验是否不是玩家角色
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    if character_id:
        return 1
    return 0


@add_premise(constant.Premise.IN_PLAYER_SCENE)
def handle_in_player_scene(character_id: int) -> int:
    """
    校验角色是否与玩家处于同场景中
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    now_character_data = cache_contorl.character_data[character_id]
    if now_character_data.position == cache_contorl.character_data[0].position:
        return 1
    return 0


@add_premise(constant.Premise.LEAVE_PLAYER_SCENE)
def handle_leave_player_scene(character_id: int) -> int:
    """
    校验角色是否是从玩家场景离开
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    now_character_data = cache_contorl.character_data[character_id]
    if (
        now_character_data.behavior["MoveSrc"]
        == cache_contorl.character_data[0].position
    ):
        return 1
    return 0


@add_premise(constant.Premise.TARGET_IS_ADORE)
def handle_target_is_adore(character_id: int) -> int:
    """
    校验角色当前目标是否是自己的爱慕对象
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id in character_data.social_contact["Adore"]:
        return 1
    return 0


@add_premise(constant.Premise.TARGET_IS_ADMIRE)
def handle_target_is_admire(character_id: int) -> int:
    """
    校验角色当前的目标是否是自己的恋慕对象
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    target_id = character_data.target_character_id
    if target_id in character_data.social_contact["Admire"]:
        return 1
    return 0


@add_premise(constant.Premise.PLAYER_IS_ADORE)
def handle_player_is_adore(character_id: int) -> int:
    """
    校验玩家是否是当前角色的爱慕对象
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    if 0 in character_data.social_contact["Adore"]:
        return 1
    return 0


@add_premise(constant.Premise.EAT_SPRING_FOOD)
def handle_eat_spring_food(character_id: int) -> int:
    """
    校验角色是否正在食用春药品质的食物
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache_contorl.character_data[character_id]
    if character_data.behavior["FoodQuality"] == 4:
        return 1
    return 0
