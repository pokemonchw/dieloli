from Script.Design import handle_premise, attr_calculation, constant
from Script.Core import game_type, cache_control

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_premise.add_premise(constant.Premise.MECHANICS_SKILLS_INTEREST_IS_HEIGHT)
def handle_mechanics_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否机械天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 32 in character_data.knowledge_interest:
        if character_data.knowledge_interest[32] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.MECHANICS_SKILLS_INTEREST_IS_LOW)
def handle_mechanics_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否机械天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 32 in character_data.knowledge_interest:
        if character_data.knowledge_interest[32] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.ELECTRONICS_SKILLS_INTEREST_IS_HEIGHT)
def handle_electronics_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否电子天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 33 in character_data.knowledge_interest:
        if character_data.knowledge_interest[33] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.ELECTRONICS_SKILLS_INTEREST_IS_LOW)
def handle_electronics_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否电子天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 33 in character_data.knowledge_interest:
        if character_data.knowledge_interest[33] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.COMPUTER_SCIENCE_SKILLS_INTEREST_IS_HEIGHT)
def handle_computer_science_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否计算机学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 34 in character_data.knowledge_interest:
        if character_data.knowledge_interest[34] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.COMPUTER_SCIENCE_SKILLS_INTEREST_IS_LOW)
def handle_computer_science_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否计算机学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 34 in character_data.knowledge_interest:
        if character_data.knowledge_interest[34] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.CRYPTOGRAPHY_SKILLS_INTEREST_IS_HEIGHT)
def handle_cryptograthy_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否密码学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 35 in character_data.knowledge_interest:
        if character_data.knowledge_interest[35] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.CRYPTOGRAPHY_SKILLS_INTEREST_IS_LOW)
def handle_cryptograthy_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否密码学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 35 in character_data.knowledge_interest:
        if character_data.knowledge_interest[35] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.CHEMISTRY_SKILLS_INTEREST_IS_HEIGHT)
def handle_chemistry_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否化学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 36 in character_data.knowledge_interest:
        if character_data.knowledge_interest[36] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.CHEMISTRY_SKILLS_INTEREST_IS_LOW)
def handle_chemistry_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否化学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 36 in character_data.knowledge_interest:
        if character_data.knowledge_interest[36] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.BIOLOGY_SKILLS_INTEREST_IS_HEIGHT)
def handle_biology_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否生物学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 37 in character_data.knowledge_interest:
        if character_data.knowledge_interest[37] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.BIOLOGY_SKILLS_INTEREST_IS_LOW)
def handle_biology_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否生物学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 37 in character_data.knowledge_interest:
        if character_data.knowledge_interest[37] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.MATHEMATICS_SKILLS_INTEREST_IS_HEIGHT)
def handle_mathematics_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否数学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 38 in character_data.knowledge_interest:
        if character_data.knowledge_interest[38] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.MATHEMATICS_SKILLS_INTEREST_IS_LOW)
def handle_mathematics_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否数学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 38 in character_data.knowledge_interest:
        if character_data.knowledge_interest[38] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.ASTRONOMY_SKILLS_INTEREST_IS_HEIGHT)
def handle_astronomy_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否天文学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 39 in character_data.knowledge_interest:
        if character_data.knowledge_interest[39] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.ASTRONOMY_SKILLS_INTEREST_IS_LOW)
def handle_astronomy_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否天文学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 39 in character_data.knowledge_interest:
        if character_data.knowledge_interest[39] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.PHYSICS_SKILLS_INTEREST_IS_HEIGHT)
def handle_physics_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否物理学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 40 in character_data.knowledge_interest:
        if character_data.knowledge_interest[40] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.PHYSICS_SKILLS_INTEREST_IS_LOW)
def handle_physics_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否物理学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 40 in character_data.knowledge_interest:
        if character_data.knowledge_interest[40] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.GEOGRAPHY_SKILLS_INTEREST_IS_HEIGHT)
def handle_geography_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否地理学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 41 in character_data.knowledge_interest:
        if character_data.knowledge_interest[41] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.GEOGRAPHY_SKILLS_INTEREST_IS_LOW)
def handle_geography_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否地理学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 41 in character_data.knowledge_interest:
        if character_data.knowledge_interest[41] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.GEOLOGY_SKILLS_INTEREST_IS_HEIGHT)
def handle_geology_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否地质学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 42 in character_data.knowledge_interest:
        if character_data.knowledge_interest[42] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.GEOLOGY_SKILLS_INTEREST_IS_LOW)
def handle_geology_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否地质学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 42 in character_data.knowledge_interest:
        if character_data.knowledge_interest[42] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.ECOLOGY_SKILLS_INTEREST_IS_HEIGHT)
def handle_ecology_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否生态学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 43 in character_data.knowledge_interest:
        if character_data.knowledge_interest[43] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.ECOLOGY_SKILLS_INTEREST_IS_LOW)
def handle_ecology_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否生态学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 43 in character_data.knowledge_interest:
        if character_data.knowledge_interest[43] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.ZOOLOGY_SKILLS_INTEREST_IS_HEIGHT)
def handle_zoology_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否动物学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 44 in character_data.knowledge_interest:
        if character_data.knowledge_interest[44] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.ZOOLOGY_SKILLS_INTEREST_IS_LOW)
def handle_zoology_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否动物学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 44 in character_data.knowledge_interest:
        if character_data.knowledge_interest[44] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.BOTANY_SKILLS_INTEREST_IS_HEIGHT)
def handle_botany_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否植物学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 45 in character_data.knowledge_interest:
        if character_data.knowledge_interest[45] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.BOTANY_SKILLS_INTEREST_IS_LOW)
def handle_botany_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否植物学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 45 in character_data.knowledge_interest:
        if character_data.knowledge_interest[45] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.ENTOMOLOGY_SKILLS_INTEREST_IS_HEIGHT)
def handle_entomology_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否昆虫学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 46 in character_data.knowledge_interest:
        if character_data.knowledge_interest[46] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.ENTOMOLOGY_SKILLS_INTEREST_IS_LOW)
def handle_entomology_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否昆虫学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 46 in character_data.knowledge_interest:
        if character_data.knowledge_interest[46] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.MICROBIOLOGY_SKILLS_INTEREST_IS_HEIGHT)
def handle_microbiology_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否微生物学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 47 in character_data.knowledge_interest:
        if character_data.knowledge_interest[47] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.MICROBIOLOGY_SKILLS_INTEREST_IS_LOW)
def handle_microbiology_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否微生物学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 47 in character_data.knowledge_interest:
        if character_data.knowledge_interest[47] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.VIROLOGY_SKILLS_INTEREST_IS_HEIGHT)
def handle_virology_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否病毒学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 48 in character_data.knowledge_interest:
        if character_data.knowledge_interest[48] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.VIROLOGY_SKILLS_INTEREST_IS_LOW)
def handle_virology_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否病毒学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 48 in character_data.knowledge_interest:
        if character_data.knowledge_interest[48] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.BECTERIOLOGY_SKILLS_INTEREST_IS_HEIGHT)
def handle_becteriology_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否细菌学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 49 in character_data.knowledge_interest:
        if character_data.knowledge_interest[49] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.BECTERIOLOGY_SKILLS_INTEREST_IS_LOW)
def handle_becteriology_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否细菌学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 49 in character_data.knowledge_interest:
        if character_data.knowledge_interest[49] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.MYCOLOGY_SKILLS_INTEREST_IS_HEIGHT)
def handle_mycology_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否真菌学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 50 in character_data.knowledge_interest:
        if character_data.knowledge_interest[50] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.MECHANICS_SKILLS_INTEREST_IS_LOW)
def handle_mycology_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否真菌学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 50 in character_data.knowledge_interest:
        if character_data.knowledge_interest[50] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.PHARMACY_SKILLS_INTEREST_IS_HEIGHT)
def handle_pharmacy_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否药学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 51 in character_data.knowledge_interest:
        if character_data.knowledge_interest[51] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.PHARMACY_SKILLS_INTEREST_IS_LOW)
def handle_pharmacy_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否药学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 51 in character_data.knowledge_interest:
        if character_data.knowledge_interest[51] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.METEOROLOGY_SKILLS_INTEREST_IS_HEIGHT)
def handle_meteorology_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否气象学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 52 in character_data.knowledge_interest:
        if character_data.knowledge_interest[52] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.METEOROLOGY_SKILLS_INTEREST_IS_LOW)
def handle_meteorology_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否气象学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 52 in character_data.knowledge_interest:
        if character_data.knowledge_interest[52] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.LAW_SCIENCE_SKILLS_INTEREST_IS_HEIGHT)
def handle_law_science_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否法学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 53 in character_data.knowledge_interest:
        if character_data.knowledge_interest[53] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.LAW_SCIENCE_SKILLS_INTEREST_IS_LOW)
def handle_law_science_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否法学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 53 in character_data.knowledge_interest:
        if character_data.knowledge_interest[53] < 1:
            return 1
    return 1


@handle_premise.add_premise(constant.Premise.LINGUISTICS_SKILLS_INTEREST_IS_HEIGHT)
def handle_linguistics_skills_interest_is_height(character_id: int) -> int:
    """
    校验角色是否语言学天赋高
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 54 in character_data.knowledge_interest:
        if character_data.knowledge_interest[54] >= 1:
            return 1
    return 0


@handle_premise.add_premise(constant.Premise.LINGUISTICS_SKILLS_INTEREST_IS_LOW)
def handle_linguistics_skills_interest_is_low(character_id: int) -> int:
    """
    校验角色是否语言学天赋低
    Keyword arguments:
    character_id -- 角色id
    Return arguments:
    int -- 权重
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if 54 in character_data.knowledge_interest:
        if character_data.knowledge_interest[54] < 1:
            return 1
    return 1
