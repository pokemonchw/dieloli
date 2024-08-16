from Script.Design import handle_premise, attr_calculation, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.TARGET_MECHANICS_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_mechanics_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否机械天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 32 in target_data.knowledge_interest:
        if target_data.knowledge_interest[32] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_MECHANICS_SKILLS_INTEREST_IS_LOW)
def handle_target_mechanics_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否机械天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 32 in target_data.knowledge_interest:
        if target_data.knowledge_interest[32] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_ELECTRONICS_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_electronics_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否电子天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 33 in target_data.knowledge_interest:
        if target_data.knowledge_interest[33] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_ELECTRONICS_SKILLS_INTEREST_IS_LOW)
def handle_target_electronics_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否电子天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 33 in target_data.knowledge_interest:
        if target_data.knowledge_interest[33] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_COMPUTER_SCIENCE_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_computer_science_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否计算机学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 34 in target_data.knowledge_interest:
        if target_data.knowledge_interest[34] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_COMPUTER_SCIENCE_SKILLS_INTEREST_IS_LOW)
def handle_target_computer_science_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否计算机学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 34 in target_data.knowledge_interest:
        if target_data.knowledge_interest[34] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_CRYPTOGRAPHY_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_cryptograthy_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否密码学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 35 in target_data.knowledge_interest:
        if target_data.knowledge_interest[35] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_CRYPTOGRAPHY_SKILLS_INTEREST_IS_LOW)
def handle_target_cryptograthy_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否密码学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 35 in target_data.knowledge_interest:
        if target_data.knowledge_interest[35] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_CHEMISTRY_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_chemistry_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否化学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 36 in target_data.knowledge_interest:
        if target_data.knowledge_interest[36] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_CHEMISTRY_SKILLS_INTEREST_IS_LOW)
def handle_target_chemistry_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否化学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 36 in target_data.knowledge_interest:
        if target_data.knowledge_interest[36] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_BIOLOGY_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_biology_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否生物学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 37 in target_data.knowledge_interest:
        if target_data.knowledge_interest[37] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_BIOLOGY_SKILLS_INTEREST_IS_LOW)
def handle_target_biology_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否生物学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 37 in target_data.knowledge_interest:
        if target_data.knowledge_interest[37] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_MATHEMATICS_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_mathematics_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否数学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 38 in target_data.knowledge_interest:
        if target_data.knowledge_interest[38] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_MATHEMATICS_SKILLS_INTEREST_IS_LOW)
def handle_target_mathematics_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否数学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 38 in target_data.knowledge_interest:
        if target_data.knowledge_interest[38] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_ASTRONOMY_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_astronomy_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否天文学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 39 in target_data.knowledge_interest:
        if target_data.knowledge_interest[39] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_ASTRONOMY_SKILLS_INTEREST_IS_LOW)
def handle_target_astronomy_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否天文学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 39 in target_data.knowledge_interest:
        if target_data.knowledge_interest[39] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_PHYSICS_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_physics_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否物理学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 40 in target_data.knowledge_interest:
        if target_data.knowledge_interest[40] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_PHYSICS_SKILLS_INTEREST_IS_LOW)
def handle_target_physics_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否物理学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 40 in target_data.knowledge_interest:
        if target_data.knowledge_interest[40] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_GEOGRAPHY_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_geography_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否地理学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 41 in target_data.knowledge_interest:
        if target_data.knowledge_interest[41] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_GEOGRAPHY_SKILLS_INTEREST_IS_LOW)
def handle_target_geography_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否地理学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 41 in target_data.knowledge_interest:
        if target_data.knowledge_interest[41] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_GEOLOGY_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_geology_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否地质学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 42 in target_data.knowledge_interest:
        if target_data.knowledge_interest[42] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_GEOLOGY_SKILLS_INTEREST_IS_LOW)
def handle_target_geology_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否地质学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 42 in target_data.knowledge_interest:
        if target_data.knowledge_interest[42] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_ECOLOGY_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_ecology_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否生态学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 43 in target_data.knowledge_interest:
        if target_data.knowledge_interest[43] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_ECOLOGY_SKILLS_INTEREST_IS_LOW)
def handle_target_ecology_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否生态学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 43 in target_data.knowledge_interest:
        if target_data.knowledge_interest[43] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_ZOOLOGY_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_zoology_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否动物学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 44 in target_data.knowledge_interest:
        if target_data.knowledge_interest[44] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_ZOOLOGY_SKILLS_INTEREST_IS_LOW)
def handle_target_zoology_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否动物学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 44 in target_data.knowledge_interest:
        if target_data.knowledge_interest[44] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_BOTANY_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_botany_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否植物学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 45 in target_data.knowledge_interest:
        if target_data.knowledge_interest[45] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_BOTANY_SKILLS_INTEREST_IS_LOW)
def handle_target_botany_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否植物学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 45 in target_data.knowledge_interest:
        if target_data.knowledge_interest[45] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_ENTOMOLOGY_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_entomology_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否昆虫学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 46 in target_data.knowledge_interest:
        if target_data.knowledge_interest[46] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_ENTOMOLOGY_SKILLS_INTEREST_IS_LOW)
def handle_target_entomology_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否昆虫学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 46 in target_data.knowledge_interest:
        if target_data.knowledge_interest[46] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_MICROBIOLOGY_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_microbiology_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否微生物学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 47 in target_data.knowledge_interest:
        if target_data.knowledge_interest[47] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_MICROBIOLOGY_SKILLS_INTEREST_IS_LOW)
def handle_target_microbiology_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否微生物学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 47 in target_data.knowledge_interest:
        if target_data.knowledge_interest[47] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_VIROLOGY_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_virology_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否病毒学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 48 in target_data.knowledge_interest:
        if target_data.knowledge_interest[48] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_VIROLOGY_SKILLS_INTEREST_IS_LOW)
def handle_target_virology_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否病毒学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 48 in target_data.knowledge_interest:
        if target_data.knowledge_interest[48] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_BECTERIOLOGY_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_becteriology_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否细菌学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 49 in target_data.knowledge_interest:
        if target_data.knowledge_interest[49] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_BECTERIOLOGY_SKILLS_INTEREST_IS_LOW)
def handle_target_becteriology_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否细菌学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 49 in target_data.knowledge_interest:
        if target_data.knowledge_interest[49] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_MYCOLOGY_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_mycology_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否真菌学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 50 in target_data.knowledge_interest:
        if target_data.knowledge_interest[50] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_MECHANICS_SKILLS_INTEREST_IS_LOW)
def handle_target_mycology_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否真菌学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 50 in target_data.knowledge_interest:
        if target_data.knowledge_interest[50] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_PHARMACY_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_pharmacy_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否药学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 51 in target_data.knowledge_interest:
        if target_data.knowledge_interest[51] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_PHARMACY_SKILLS_INTEREST_IS_LOW)
def handle_target_pharmacy_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否药学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 51 in target_data.knowledge_interest:
        if target_data.knowledge_interest[51] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_METEOROLOGY_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_meteorology_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否气象学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 52 in target_data.knowledge_interest:
        if target_data.knowledge_interest[52] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_METEOROLOGY_SKILLS_INTEREST_IS_LOW)
def handle_target_meteorology_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否气象学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 52 in target_data.knowledge_interest:
        if target_data.knowledge_interest[52] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_LAW_SCIENCE_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_law_science_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否法学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 53 in target_data.knowledge_interest:
        if target_data.knowledge_interest[53] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_LAW_SCIENCE_SKILLS_INTEREST_IS_LOW)
def handle_target_law_science_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否法学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 53 in target_data.knowledge_interest:
        if target_data.knowledge_interest[53] < 1:
            return 1
        return 0
    return 1


@handle_premise.add_premise(constant.Premise.TARGET_LINGUISTICS_SKILLS_INTEREST_IS_HEIGHT)
def handle_target_linguistics_skills_interest_is_height(character_id: int) -> int:
    """
    校验交互对象是否语言学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 54 in target_data.knowledge_interest:
        if target_data.knowledge_interest[54] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.TARGET_LINGUISTICS_SKILLS_INTEREST_IS_LOW)
def handle_target_linguistics_skills_interest_is_low(character_id: int) -> int:
    """
    校验交互对象是否语言学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.target_character_id == -1:
        return 0
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    if 54 in target_data.knowledge_interest:
        if target_data.knowledge_interest[54] < 1:
            return 1
        return 0
    return 1
