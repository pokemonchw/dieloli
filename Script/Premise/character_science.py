from Script.Design import handle_premise, attr_calculation, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.MECHANICS_SKILLS_IS_HEIGHT)
def handle_mechanics_skills_is_height(character_id: int) -> int:
    """
    校验角色是否机械水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[32]
    if 32 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[32])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.MECHANICS_SKILLS_IS_LOW)
def handle_mechanics_skills_is_low(character_id: int) -> int:
    """
    校验角色是否机械水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 32 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[32])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.ELECTRONICS_SKILLS_IS_HEIGHT)
def handle_electronics_skills_is_height(character_id: int) -> int:
    """
    校验角色是否电子水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[33]
    if 33 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[33])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.ELECTRONICS_SKILLS_IS_LOW)
def handle_electronics_skills_is_low(character_id: int) -> int:
    """
    校验角色是否电子水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 33 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[33])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.COMPUTER_SCIENCE_SKILLS_IS_HEIGHT)
def handle_computer_science_skills_is_height(character_id: int) -> int:
    """
    校验角色是否计算机学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[34]
    if 34 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[34])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.COMPUTER_SCIENCE_SKILLS_IS_LOW)
def handle_computer_science_skills_is_low(character_id: int) -> int:
    """
    校验角色是否计算机学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 34 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[34])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.CRYPTOGRAPHY_SKILLS_IS_HEIGHT)
def handle_cryptograthy_skills_is_height(character_id: int) -> int:
    """
    校验角色是否密码学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[35]
    if 35 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[35])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.CRYPTOGRAPHY_SKILLS_IS_LOW)
def handle_cryptograthy_skills_is_low(character_id: int) -> int:
    """
    校验角色是否密码学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 35 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[35])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.CHEMISTRY_SKILLS_IS_HEIGHT)
def handle_chemistry_skills_is_height(character_id: int) -> int:
    """
    校验角色是否化学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[36]
    if 36 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[36])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.CHEMISTRY_SKILLS_IS_LOW)
def handle_chemistry_skills_is_low(character_id: int) -> int:
    """
    校验角色是否化学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 36 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[36])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.BIOLOGY_SKILLS_IS_HEIGHT)
def handle_biology_skills_is_height(character_id: int) -> int:
    """
    校验角色是否生物学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[37]
    if 37 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[37])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.BIOLOGY_SKILLS_IS_LOW)
def handle_biology_skills_is_low(character_id: int) -> int:
    """
    校验角色是否生物学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 37 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[37])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.MATHEMATICS_SKILLS_IS_HEIGHT)
def handle_mathematics_skills_is_height(character_id: int) -> int:
    """
    校验角色是否数学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[38]
    if 38 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[38])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.MATHEMATICS_SKILLS_IS_LOW)
def handle_mathematics_skills_is_low(character_id: int) -> int:
    """
    校验角色是否数学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 38 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[38])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.ASTRONOMY_SKILLS_IS_HEIGHT)
def handle_astronomy_skills_is_height(character_id: int) -> int:
    """
    校验角色是否天文学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[39]
    if 39 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[39])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.ASTRONOMY_SKILLS_IS_LOW)
def handle_astronomy_skills_is_low(character_id: int) -> int:
    """
    校验角色是否天文学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 39 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[39])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.PHYSICS_SKILLS_IS_HEIGHT)
def handle_physics_skills_is_height(character_id: int) -> int:
    """
    校验角色是否物理学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[40]
    if 40 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[40])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.PHYSICS_SKILLS_IS_LOW)
def handle_physics_skills_is_low(character_id: int) -> int:
    """
    校验角色是否物理学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 40 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[40])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.GEOGRAPHY_SKILLS_IS_HEIGHT)
def handle_geography_skills_is_height(character_id: int) -> int:
    """
    校验角色是否地理学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[41]
    if 41 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[41])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.GEOGRAPHY_SKILLS_IS_LOW)
def handle_geography_skills_is_low(character_id: int) -> int:
    """
    校验角色是否地理学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 41 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[41])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.GEOLOGY_SKILLS_IS_HEIGHT)
def handle_geology_skills_is_height(character_id: int) -> int:
    """
    校验角色是否地质学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[42]
    if 42 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[42])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.GEOLOGY_SKILLS_IS_LOW)
def handle_geology_skills_is_low(character_id: int) -> int:
    """
    校验角色是否地质学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 42 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[42])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.ECOLOGY_SKILLS_IS_HEIGHT)
def handle_ecology_skills_is_height(character_id: int) -> int:
    """
    校验角色是否生态学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[43]
    if 43 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[43])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.ECOLOGY_SKILLS_IS_LOW)
def handle_ecology_skills_is_low(character_id: int) -> int:
    """
    校验角色是否生态学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 43 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[43])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.ZOOLOGY_SKILLS_IS_HEIGHT)
def handle_zoology_skills_is_height(character_id: int) -> int:
    """
    校验角色是否动物学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[44]
    if 44 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[44])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.ZOOLOGY_SKILLS_IS_LOW)
def handle_zoology_skills_is_low(character_id: int) -> int:
    """
    校验角色是否动物学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 44 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[44])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.BOTANY_SKILLS_IS_HEIGHT)
def handle_botany_skills_is_height(character_id: int) -> int:
    """
    校验角色是否植物学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[45]
    if 45 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[45])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.BOTANY_SKILLS_IS_LOW)
def handle_botany_skills_is_low(character_id: int) -> int:
    """
    校验角色是否植物学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 45 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[45])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.ENTOMOLOGY_SKILLS_IS_HEIGHT)
def handle_entomology_skills_is_height(character_id: int) -> int:
    """
    校验角色是否昆虫学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[46]
    if 46 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[46])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.ENTOMOLOGY_SKILLS_IS_LOW)
def handle_entomology_skills_is_low(character_id: int) -> int:
    """
    校验角色是否昆虫学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 46 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[46])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.MICROBIOLOGY_SKILLS_IS_HEIGHT)
def handle_microbiology_skills_is_height(character_id: int) -> int:
    """
    校验角色是否微生物学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[47]
    if 47 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[47])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.MICROBIOLOGY_SKILLS_IS_LOW)
def handle_microbiology_skills_is_low(character_id: int) -> int:
    """
    校验角色是否微生物学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 47 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[47])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.VIROLOGY_SKILLS_IS_HEIGHT)
def handle_virology_skills_is_height(character_id: int) -> int:
    """
    校验角色是否病毒学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[48]
    if 48 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[48])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.VIROLOGY_SKILLS_IS_LOW)
def handle_virology_skills_is_low(character_id: int) -> int:
    """
    校验角色是否病毒学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 48 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[48])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.BECTERIOLOGY_SKILLS_IS_HEIGHT)
def handle_becteriology_skills_is_height(character_id: int) -> int:
    """
    校验角色是否细菌学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[49]
    if 49 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[49])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.BECTERIOLOGY_SKILLS_IS_LOW)
def handle_becteriology_skills_is_low(character_id: int) -> int:
    """
    校验角色是否细菌学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 49 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[49])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.MYCOLOGY_SKILLS_IS_HEIGHT)
def handle_mycology_skills_is_height(character_id: int) -> int:
    """
    校验角色是否真菌学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[50]
    if 50 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[50])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.MECHANICS_SKILLS_IS_LOW)
def handle_mycology_skills_is_low(character_id: int) -> int:
    """
    校验角色是否真菌学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 50 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[50])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.PHARMACY_SKILLS_IS_HEIGHT)
def handle_pharmacy_skills_is_height(character_id: int) -> int:
    """
    校验角色是否药学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[51]
    if 51 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[51])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.PHARMACY_SKILLS_IS_LOW)
def handle_pharmacy_skills_is_low(character_id: int) -> int:
    """
    校验角色是否药学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 51 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[51])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.METEOROLOGY_SKILLS_IS_HEIGHT)
def handle_meteorology_skills_is_height(character_id: int) -> int:
    """
    校验角色是否气象学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[52]
    if 52 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[52])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.METEOROLOGY_SKILLS_IS_LOW)
def handle_meteorology_skills_is_low(character_id: int) -> int:
    """
    校验角色是否气象学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 52 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[52])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.LAW_SCIENCE_SKILLS_IS_HEIGHT)
def handle_law_science_skills_is_height(character_id: int) -> int:
    """
    校验角色是否法学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[53]
    if 53 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[53])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.LAW_SCIENCE_SKILLS_IS_LOW)
def handle_law_science_skills_is_low(character_id: int) -> int:
    """
    校验角色是否法学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 53 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[53])
        if level <= 2:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.LINGUISTICS_SKILLS_IS_HEIGHT)
def handle_linguistics_skills_is_height(character_id: int) -> int:
    """
    校验角色是否语言学水平高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    weight = 1 + character_data.knowledge_interest[54]
    if 54 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[54])
        return weight * level
    return weight


@handle_premise.add_premise(constant.Premise.LINGUISTICS_SKILLS_IS_LOW)
def handle_linguistics_skills_is_low(character_id: int) -> int:
    """
    校验角色是否语言学水平低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 54 in character_data.knowledge:
        level = attr_calculation.get_experience_level_weight(character_data.knowledge[54])
        if level <= 2:
            return 1
        return 0
    return 1
