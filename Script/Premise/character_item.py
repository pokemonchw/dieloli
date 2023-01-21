from Script.Design import handle_premise, constant
from Script.Core import game_type, cache_control
from Script.Config import game_config

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.HAVE_FOOD)
def handle_have_food(character_id: int) -> int:
    """
    校验角色是否拥有食物
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    food_index = 0
    for food_id in character_data.food_bag:
        if character_data.food_bag[food_id].eat and 27 in character_data.food_bag[food_id].feel:
            food_index += 1
    return food_index


@handle_premise.add_premise(constant.Premise.NOT_HAVE_FOOD)
def handle_not_have_food(character_id: int) -> int:
    """
    校验角色是否没有食物
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    food_index = 1
    for food_id in character_data.food_bag:
        if character_data.food_bag[food_id].eat and 27 in character_data.food_bag[food_id].feel:
            return 0
    return food_index


@handle_premise.add_premise(constant.Premise.HAVE_DRAW_ITEM)
def handle_have_item_by_tag_draw(character_id: int) -> int:
    """
    校验角色是否拥有绘画类道具
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    for item_id in game_config.config_item_tag_data["Draw"]:
        if item_id in character_data.item:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.HAVE_SHOOTING_ITEM)
def handle_have_item_by_tag_shooting(character_id: int) -> int:
    """
    校验角色是否拥有射击类道具
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    for item_id in game_config.config_item_tag_data["Shooting"]:
        if item_id in character_data.item:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.HAVE_GUITAR)
def handle_have_guitar(character_id: int) -> int:
    """
    校验角色是否拥有吉他
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    if 4 in character_data.item:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.NO_HAVE_GUITAR)
def handle_no_have_guitar(character_id: int) -> int:
    """
    校验角色是否未拥有吉他
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    return 4 not in character_data.item


@handle_premise.add_premise(constant.Premise.HAVE_HARMONICA)
def handle_have_harmonica(character_id: int) -> int:
    """
    校验角色是否拥有口琴
    Keyword arguments:
    character_id --角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    if 5 in character_data.item:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.HAVE_BAM_BOO_FLUTE)
def handle_have_bamboogflute(character_id: int) -> int:
    """
    校验角色是否拥有竹笛
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    if 6 in character_data.item:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.HAVE_BASKETBALL)
def handle_have_basketball(character_id: int) -> int:
    """
    校验角色是否拥有篮球
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    if 0 in character_data.item:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.HAVE_FOOTBALL)
def handle_have_football(character_id: int) -> int:
    """
    校验角色是否拥有足球
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    if 1 in character_data.item:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.HAVE_TABLE_TENNIS)
def handle_have_tabletennis(character_id: int) -> int:
    """
    校验角色是否拥有乒乓球
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    if 2 in character_data.item:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.HAVE_UNDERWEAR)
def handle_have_underwear(character_id: int) -> int:
    """
    校验角色是否拥有上衣
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 1 in character_data.clothing and character_data.clothing[1]:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.HAVE_UNDERPANTS)
def handle_have_underpants(character_id: int) -> int:
    """
    校验角色是否拥有内裤
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 7 in character_data.clothing and character_data.clothing[7]:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.HAVE_BRA)
def handle_have_bra(character_id: int) -> int:
    """
    校验角色是否拥有胸罩
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 6 in character_data.clothing and character_data.clothing[6]:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.HAVE_PANTS)
def handle_have_pants(character_id: int) -> int:
    """
    校验角色是否拥有裤子
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 2 in character_data.clothing and character_data.clothing[2]:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.HAVE_SKIRT)
def handle_have_skirt(character_id: int) -> int:
    """
    校验角色是否拥有短裙
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 3 in character_data.clothing and character_data.clothing[3]:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.HAVE_SHOES)
def handle_have_shoes(character_id: int) -> int:
    """
    校验角色是否拥有鞋子
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 4 in character_data.clothing and character_data.clothing[4]:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.HAVE_SOCKS)
def handle_have_socks(character_id: int) -> int:
    """
    校验角色是否拥有袜子
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 5 in character_data.clothing and character_data.clothing[5]:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.HAVE_DRINKS)
def handle_have_drinks(character_id: int) -> int:
    """
    校验角色背包中是否有饮料
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    drinks_list = []
    for food_id in character_data.food_bag:
        now_food: game_type.Food = character_data.food_bag[food_id]
        if now_food.eat and 28 in now_food.feel:
            drinks_list.append(food_id)
    return len(drinks_list)


@handle_premise.add_premise(constant.Premise.NO_HAVE_DRINKS)
def handle_no_have_drinks(character_id: int) -> int:
    """
    校验角色背包中是否没有饮料
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    for food_id in character_data.food_bag:
        now_food: game_type.Food = character_data.food_bag[food_id]
        if now_food.eat and 28 in now_food.feel:
            return 0
    return 1


@handle_premise.add_premise(constant.Premise.HAVE_COAT)
def handle_have_coat(character_id: int) -> int:
    """
    校验角色是否拥有外套
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 0 in character_data.clothing and character_data.clothing[0]:
        return 1
    return 0
