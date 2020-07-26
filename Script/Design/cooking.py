import random
import uuid
from typing import Dict
from Script.Core.game_type import Recipes, Food
from Script.Core import constant, text_loading, cache_contorl, value_handle


def init_recipes():
    """ 初始化菜谱数据 """
    recipes_data = text_loading.get_game_data(constant.FilePath.RECIPES_PATH)
    cache_contorl.recipe_data = {}
    for recipe_dict in recipes_data:
        recipe = create_recipe(
            recipe_dict["Name"],
            recipe_dict["Time"],
            recipe_dict["Base"],
            recipe_dict["Ingredients"],
            recipe_dict["Seasoning"],
        )
        cache_contorl.recipe_data[len(cache_contorl.recipe_data)] = recipe


def create_recipe(
    name: str, time: int, base: list, ingredients: list, seasoning: list
) -> Recipes:
    """
    创建菜谱对象
    Keyword arguments:
    name -- 菜谱名字
    time -- 烹饪用时
    base -- 主食材列表
    ingredients -- 辅食材列表
    seasoning -- 调料列表
    Return arguments:
    Recipes -- 菜谱对象
    """
    recipe = Recipes()
    recipe.name = name
    recipe.time = time
    recipe.base = base
    recipe.ingredients = ingredients
    recipe.seasoning = seasoning
    return recipe


def create_food(
    food_id: str,
    food_quality: int,
    food_weight: int,
    food_feel={},
    food_maker="",
    food_recipe=-1,
) -> Food:
    """
    创建食物对象
    Keyword arguments:
    food_id -- 食物配置id
    food_quality -- 食物品质
    food_weight -- 食物重量
    food_feel -- 食物效果
    food_maker -- 食物制作者
    food_recipe -- 食谱id(为-1时表示不是用食谱制作出来的基础食材)
    Return arguments:
    Food -- 食物对象
    """
    food = Food()
    food.id = food_id
    food.uid = uuid.uuid4()
    food.quality = food_quality
    food.weight = food_weight
    if food_id != "" and food_feel == {}:
        food_config = text_loading.get_text_data(
            constant.FilePath.FOOD_PATH, food_id
        )
        for feel in food_config["Feel"]:
            food.feel.setdefault(feel, 0)
            food.feel[feel] += food_config["Feel"][feel] / 100 * food.weight
        food.cook = food_config["Cook"]
        food.eat = food_config["Eat"]
        food.seasoning = food_config["Seasoning"]
        food.fruit = food_config["Fruit"]
    else:
        food.feel = food_feel
        food.eat = 1
        food.cook = 0
        food.seasoning = 0
    food.maker = food_maker
    food.recipe = food_recipe
    return food


def separate_weight_food(old_food: Food, weight: int) -> Food:
    """
    从指定食物中分离出指定重量的食物并创建新食物对象
    Keyword arguments:
    old_food -- 原本的食物对象
    Return arguments:
    Food -- 分离得到的食物对象
    """
    new_food = Food()
    if old_food.weight < weight:
        weight = old_food.weight
    new_food.cook = old_food.cook
    new_food.eat = old_food.eat
    new_food.fruit = old_food.fruit
    new_food.id = old_food.id
    new_food.uid = uuid.uuid4()
    new_food.maker = old_food.maker
    new_food.quality = old_food.quality
    new_food.recipe = old_food.recipe
    new_food.seasoning = old_food.seasoning
    new_food.weight = weight
    for feel in old_food.feel:
        now_feel_num = old_food.feel[feel] / old_food.weight * weight
        new_food.feel[feel] = now_feel_num
        old_food.feel[feel] -= now_feel_num
    return new_food


def create_rand_food(food_id: str, food_weight=-1, food_quality=-1) -> Food:
    """
    创建随机食材
    Keyword arguments:
    food_id -- 食物配置id
    food_weight -- 食物重量(为-1时随机)
    food_quality -- 食物品质(为-1时随机)
    Return arguments:
    Food -- 食物对象
    """
    if food_weight == -1:
        food_weight = random.randint(1, 1000000)
    if food_quality == -1:
        food_quality = random.randint(0, 4)
    food_data = text_loading.get_game_data(constant.FilePath.FOOD_PATH)[
        food_id
    ]
    return create_food(food_id, food_quality, food_weight)


def cook(
    food_data: Dict[str, Food], recipe_id: int, cook_level: str, maker: str
) -> Food:
    """
    按食谱烹饪食物
    Keyword arguments:
    food_data -- 食材数据
    recipe_id -- 菜谱id
    cook_level -- 烹饪技能等级
    maker -- 制作者
    Return arguments:
    Food -- 食物对象
    """
    recipe = cache_contorl.recipe_data[recipe_id]
    cook_judge = True
    feel_data = {}
    quality_data = text_loading.get_text_data(
        constant.FilePath.ATTR_TEMPLATE_PATH, "FoodQualityWeight"
    )[str(cook_level)]
    now_quality = int(value_handle.get_random_for_weight(quality_data))
    now_weight = 0
    for food in recipe.base:
        if food not in food_data:
            cook_judge = False
            break
        now_food = food_data[food]
        rand_weight = random.randint(75, 125)
        if now_food.weight < rand_weight:
            cook_judge = False
            break
        for feel in now_food.feel:
            feel_data.setdefault(feel, 0)
            feel_data[feel] += (
                now_food.feel[feel] / now_food.weight * rand_weight
            )
        now_food.weight -= rand_weight
        now_weight += rand_weight
    if not cook_judge:
        return create_food("KitchenWaste", now_quality, now_weight, [])
    for food in recipe.ingredients:
        if food not in food_data:
            cook_judge = False
            break
        now_food = food_data[food]
        rand_weight = random.randint(25, 75)
        if now_food.weight < rand_weight:
            cook_judge = False
            break
        for feel in now_food.feel:
            feel_data.setdefault(feel, 0)
            feel_data[feel] += (
                now_food.feel[feel] / now_food.weight * rand_weight
            )
        now_food.weight -= rand_weight
        now_weight += rand_weight
    if not cook_judge:
        return create_food("KitchenWaste", now_quality, now_weight, [])
    for food in recipe.seasoning:
        if food not in food_data:
            cook_judge = False
            break
        now_food = food_data[food]
        rand_weight = random.randint(3, 7)
        if now_food.weight < rand_weight:
            cook_judge = False
            break
        for feel in now_food.feel:
            feel_data.setdefault(feel, 0)
            now_feel_num = now_food.feel[feel] / now_food.weight * rand_weight
            feel_data[feel] += now_feel_num
            now_food.feel[feel] -= now_feel_num
        now_food.weight -= rand_weight
        now_weight += rand_weight
    if not cook_judge:
        return create_food("KitchenWaste", now_quality, now_weight, [])
    return create_food(
        "", now_quality, now_weight, feel_data, maker, recipe_id
    )


def init_restaurant_data():
    """ 初始化餐馆内的食物数据 """
    cache_contorl.restaurant_data = {}
    max_people = len(cache_contorl.character_data)
    food_config_data = text_loading.get_game_data(constant.FilePath.FOOD_PATH)
    for food_id in food_config_data:
        food = create_rand_food(food_id, max_people * 100)
        cache_contorl.restaurant_data.setdefault(food_id, {})
        cache_contorl.restaurant_data[food_id][food.uid] = food
    cook_index = 0
    while 1:
        recipes_id = random.randint(0, len(cache_contorl.recipe_data) - 1)
        food_list = {}
        recipes = cache_contorl.recipe_data[recipes_id]
        food_judge = True
        for food_id in recipes.base:
            if food_id not in cache_contorl.restaurant_data or not len(
                cache_contorl.restaurant_data[food_id]
            ):
                food_judge = False
                break
            food_id_list = list(cache_contorl.restaurant_data[food_id].keys())
            now_food_id = food_id_list[0]
            now_food = cache_contorl.restaurant_data[food_id][now_food_id]
            food_list[now_food.id] = now_food
        if not food_judge:
            continue
        for food_id in recipes.ingredients:
            if food_id not in cache_contorl.restaurant_data or not len(
                cache_contorl.restaurant_data[food_id]
            ):
                food_judge = False
                break
            food_id_list = list(cache_contorl.restaurant_data[food_id].keys())
            now_food_id = food_id_list[0]
            now_food = cache_contorl.restaurant_data[food_id][now_food_id]
            food_list[now_food.id] = now_food
        if not food_judge:
            continue
        for food_id in recipes.seasoning:
            if food_id not in cache_contorl.restaurant_data or not len(
                cache_contorl.restaurant_data[food_id]
            ):
                food_judge = False
                break
            food_id_list = list(cache_contorl.restaurant_data[food_id].keys())
            now_food_id = food_id_list[0]
            now_food = cache_contorl.restaurant_data[food_id][now_food_id]
            food_list[now_food.id] = now_food
        if not food_judge:
            continue
        new_food = cook(food_list, recipes_id, 5, "")
        cache_contorl.restaurant_data.setdefault(recipes_id, {})
        cache_contorl.restaurant_data[recipes_id][new_food.uid] = new_food
        for food_id in food_list:
            now_food = food_list[food_id]
            if now_food.weight <= 0:
                del cache_contorl.restaurant_data[now_food.id][now_food.uid]
            else:
                cache_contorl.restaurant_data[now_food.id][
                    now_food.uid
                ] = now_food
        cook_index += 1
        if cook_index == max_people:
            break


def get_restaurant_food_type_list_buy_food_type(food_type: str) -> dict:
    """
    获取餐馆内指定类型的食物列表
    Keyword arguments:
    food_type -- 食物类型
    Return arguments:
    dict -- 食物列表 食物id:食物名字
    """
    food_list = {}
    for food_id in cache_contorl.restaurant_data:
        if food_type == "StapleFood":
            if isinstance(food_id, int):
                food_list[food_id] = cache_contorl.recipe_data[food_id].name
        elif food_type == "Snacks":
            if isinstance(food_id, str):
                food_config = text_loading.get_text_data(
                    constant.FilePath.FOOD_PATH, food_id
                )
                if food_config["Eat"]:
                    food_list[food_id] = food_config["Name"]
        elif food_type == "Drink":
            if isinstance(food_id, str):
                food_config = text_loading.get_text_data(
                    constant.FilePath.FOOD_PATH, food_id
                )
                if (
                    "Thirsty" in food_config["Feel"]
                    and not food_config["Fruit"]
                    and food_config["Eat"]
                    and (
                        "Hunger" not in food_config["Feel"]
                        or food_config["Feel"]["Thirsty"]
                        > food_config["Feel"]["Hunger"]
                    )
                ):
                    food_list[food_id] = food_config["Name"]
            else:
                now_food_uid = list(
                    cache_contorl.restaurant_data[food_id].keys()
                )[0]
                now_food = cache_contorl.restaurant_data[food_id][now_food_uid]
                if "Thirsty" in now_food.feel and (
                    "Hunger" not in now_food.feel
                    or now_food.feel["Thirsty"] > now_food.feel["Hunger"]
                ):
                    food_list[food_id] = cache_contorl.recipe_data[
                        food_id
                    ].name
        elif food_type == "Fruit":
            if isinstance(food_id, str):
                food_config = text_loading.get_text_data(
                    constant.FilePath.FOOD_PATH, food_id
                )
                if food_config["Fruit"]:
                    food_list[food_id] = food_config["Name"]
        elif food_type == "FoodIngredients":
            if isinstance(food_id, str):
                food_config = text_loading.get_text_data(
                    constant.FilePath.FOOD_PATH, food_id
                )
                if food_config["Cook"]:
                    food_list[food_id] = food_config["Name"]
        elif food_type == "Seasoning":
            if isinstance(food_id, str):
                food_config = text_loading.get_text_data(
                    constant.FilePath.FOOD_PATH, food_id
                )
                if food_config["Seasoning"]:
                    food_list[food_id] = food_config["Name"]
    return food_list
