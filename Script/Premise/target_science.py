from Script.Design import handle_premise, attr_calculation, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.TARGET_MECHANICS_SKILLS_IS_HEIGHT)
def handle_target_mechanics_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否机械水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 32 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[32])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_MECHANICS_SKILLS_IS_LOW)
def handle_target_mechanics_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否机械水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 32 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[32])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_ELECTRONICS_SKILLS_IS_HEIGHT)
def handle_target_electronics_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否电子水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 33 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[33])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_ELECTRONICS_SKILLS_IS_LOW)
def handle_target_electronics_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否电子水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 33 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[33])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_COMPUTER_SCIENCE_SKILLS_IS_HEIGHT)
def handle_target_computer_science_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否计算机学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 34 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[34])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_COMPUTER_SCIENCE_SKILLS_IS_LOW)
def handle_target_computer_science_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否计算机学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 34 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[34])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_CRYPTOGRAPHY_SKILLS_IS_HEIGHT)
def handle_target_cryptograthy_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否密码学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 35 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[35])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_CRYPTOGRAPHY_SKILLS_IS_LOW)
def handle_target_cryptograthy_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否密码学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 35 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[35])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_CHEMISTRY_SKILLS_IS_HEIGHT)
def handle_target_chemistry_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否化学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 36 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[36])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_CHEMISTRY_SKILLS_IS_LOW)
def handle_target_chemistry_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否化学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 36 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[36])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_BIOLOGY_SKILLS_IS_HEIGHT)
def handle_target_biology_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否生物学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 37 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[37])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_BIOLOGY_SKILLS_IS_LOW)
def handle_target_biology_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否生物学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 37 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[37])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_MATHEMATICS_SKILLS_IS_HEIGHT)
def handle_target_mathematics_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否数学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 38 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[38])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_MATHEMATICS_SKILLS_IS_LOW)
def handle_target_mathematics_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否数学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 38 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[38])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_ASTRONOMY_SKILLS_IS_HEIGHT)
def handle_target_astronomy_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否天文学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 39 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[39])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_ASTRONOMY_SKILLS_IS_LOW)
def handle_target_astronomy_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否天文学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 39 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[39])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_PHYSICS_SKILLS_IS_HEIGHT)
def handle_target_physics_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否物理学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 40 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[40])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_PHYSICS_SKILLS_IS_LOW)
def handle_target_physics_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否物理学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 40 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[40])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_GEOGRAPHY_SKILLS_IS_HEIGHT)
def handle_target_geography_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否地理学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 41 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[41])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_GEOGRAPHY_SKILLS_IS_LOW)
def handle_target_geography_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否地理学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 41 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[41])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_GEOLOGY_SKILLS_IS_HEIGHT)
def handle_target_geology_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否地质学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 42 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[42])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_GEOLOGY_SKILLS_IS_LOW)
def handle_target_geology_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否地质学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 42 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[42])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_ECOLOGY_SKILLS_IS_HEIGHT)
def handle_target_ecology_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否生态学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 43 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[43])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_ECOLOGY_SKILLS_IS_LOW)
def handle_target_ecology_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否生态学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 43 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[43])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_ZOOLOGY_SKILLS_IS_HEIGHT)
def handle_target_zoology_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否动物学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 44 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[44])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_ZOOLOGY_SKILLS_IS_LOW)
def handle_target_zoology_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否动物学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 44 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[44])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_BOTANY_SKILLS_IS_HEIGHT)
def handle_target_botany_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否植物学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 45 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[45])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_BOTANY_SKILLS_IS_LOW)
def handle_target_botany_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否植物学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 45 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[45])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_ENTOMOLOGY_SKILLS_IS_HEIGHT)
def handle_target_entomology_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否昆虫学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 46 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[46])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_ENTOMOLOGY_SKILLS_IS_LOW)
def handle_target_entomology_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否昆虫学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 46 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[46])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_MICROBIOLOGY_SKILLS_IS_HEIGHT)
def handle_target_microbiology_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否微生物学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 47 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[47])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_MICROBIOLOGY_SKILLS_IS_LOW)
def handle_target_microbiology_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否微生物学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 47 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[47])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_VIROLOGY_SKILLS_IS_HEIGHT)
def handle_target_virology_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否病毒学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 48 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[48])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_VIROLOGY_SKILLS_IS_LOW)
def handle_target_virology_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否病毒学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 48 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[48])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_BECTERIOLOGY_SKILLS_IS_HEIGHT)
def handle_target_becteriology_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否细菌学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 49 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[49])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_BECTERIOLOGY_SKILLS_IS_LOW)
def handle_target_becteriology_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否细菌学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 49 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[49])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_MYCOLOGY_SKILLS_IS_HEIGHT)
def handle_target_mycology_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否真菌学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 50 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[50])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_MECHANICS_SKILLS_IS_LOW)
def handle_target_mycology_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否真菌学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 50 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[50])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_PHARMACY_SKILLS_IS_HEIGHT)
def handle_target_pharmacy_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否药学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 51 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[51])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_PHARMACY_SKILLS_IS_LOW)
def handle_target_pharmacy_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否药学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 51 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[51])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_METEOROLOGY_SKILLS_IS_HEIGHT)
def handle_target_meteorology_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否气象学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 52 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[52])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_METEOROLOGY_SKILLS_IS_LOW)
def handle_target_meteorology_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否气象学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 52 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[52])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_LAW_SCIENCE_SKILLS_IS_HEIGHT)
def handle_target_law_science_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否法学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 53 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[53])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_LAW_SCIENCE_SKILLS_IS_LOW)
def handle_target_law_science_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否法学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 53 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[53])
        return level < 3
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_LINGUISTICS_SKILLS_IS_HEIGHT)
def handle_target_linguistics_skills_is_height(character_id: int) -> int:
    """
    校验交互对象是否语言学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 54 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[54])
        return level > 5
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_LINGUISTICS_SKILLS_IS_LOW)
def handle_target_linguistics_skills_is_low(character_id: int) -> int:
    """
    校验交互对象是否语言学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 54 in target_data.knowledge:
        level = attr_calculation.get_experience_level_weight(target_data.knowledge[54])
        return level < 3
    return 1
