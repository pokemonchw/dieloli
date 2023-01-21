from Script.Design import handle_premise, attr_calculation, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.TARGET_AGE_SIMILAR)
def handle_target_age_similar(character_id: int) -> int:
    """
    校验角色目标对像是否与自己年龄相差不大
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    target_data = cache.character_data[character_data.target_character_id]
    if character_data.age >= target_data.age - 2 and character_data.age <= target_data.age + 2:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_AVERAGE_HEIGHT_SIMILAR)
def handle_target_average_height_similar(character_id: int) -> int:
    """
    校验角色目标身高是否与平均身高相差不大
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    target_data = cache.character_data[character_data.target_character_id]
    age_tem = attr_calculation.judge_age_group(target_data.age)
    average_height = cache.average_height_by_age[age_tem][target_data.sex]
    if (
        target_data.height.now_height >= average_height * 0.95
        and target_data.height.now_height <= average_height * 1.05
    ):
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_AVERAGE_HEIGHT_LOW)
def handle_target_average_height_low(character_id: int) -> int:
    """
    校验角色目标的身高是否低于平均身高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    target_data = cache.character_data[character_data.target_character_id]
    age_tem = attr_calculation.judge_age_group(target_data.age)
    average_height = cache.average_height_by_age[age_tem][target_data.sex]
    if target_data.height.now_height <= average_height * 0.95:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_AVERGAE_STATURE_SIMILAR)
def handle_target_average_stature_similar(character_id: int) -> int:
    """
    校验角色目体型高是否与平均体型相差不大
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    target_data = cache.character_data[character_data.target_character_id]
    age_tem = attr_calculation.judge_age_group(target_data.age)
    if age_tem in cache.average_bodyfat_by_age:
        average_bodyfat = cache.average_bodyfat_by_age[age_tem][target_data.sex]
        if (
            target_data.bodyfat >= average_bodyfat * 0.95
            and target_data.bodyfat <= average_bodyfat * 1.05
        ):
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.CHEST_IS_NOT_CLIFF)
def handle_chest_is_not_cliff(character_id: int) -> int:
    """
    校验角色胸围是否不是绝壁
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    return attr_calculation.judge_chest_group(character_data.chest.now_chest)


@handle_premise.add_premise(constant.Premise.TARGET_HEIGHT_LOW)
def handle_target_height_low(character_id: int) -> int:
    """
    校验交互对象身高是否低于自身身高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.height.now_height < character_data.height.now_height:
        return character_data.height.now_height - target_data.height.now_height
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_IS_HEIGHT)
def handle_target_is_height(character_id: int) -> int:
    """
    校验角色目标身高是否高于自身身高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if target_data.height.now_height >= character_data.height.now_height * 1.05:
        return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_CHEST_IS_CLIFF)
def handle__target_chest_is_cliff(character_id: int) -> int:
    """
    校验交互对象胸围是否是绝壁
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    return not attr_calculation.judge_chest_group(target_data.chest.now_chest)


@handle_premise.add_premise(constant.Premise.TARGET_AVERAGE_STATURE_HEIGHT)
def handle_target_average_stature_height(character_id: int) -> int:
    """
    校验角色交互对象体型是否比平均体型更胖
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data = cache.character_data[character_id]
    target_data = cache.character_data[character_data.target_character_id]
    age_tem = attr_calculation.judge_age_group(target_data.age)
    if age_tem in cache.average_bodyfat_by_age:
        average_bodyfat = cache.average_bodyfat_by_age[age_tem][target_data.sex]
        if target_data.bodyfat > average_bodyfat * 1.05:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.IS_AVERAGE_STATURE_LOW)
def handle_is_average_stature_low(character_id: int) -> int:
    """
    校验角色体型是否瘦于平均体型
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    age_tem = attr_calculation.judge_age_group(character_data.age)
    if age_tem in cache.average_bodyfat_by_age:
        average_bodyfat = cache.average_bodyfat_by_age[age_tem][character_data.sex]
        if character_data.bodyfat < average_bodyfat * 0.95:
            return 1
    return 0
