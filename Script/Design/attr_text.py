import os
import random
import bisect
from Script.Core import (
    text_loading,
    cache_contorl,
    game_config,
    game_path_config,
    json_handle,
    constant,
)
from Script.Design import (
    proportional_bar,
    attr_print,
    attr_calculation,
    map_handle,
)

language = game_config.language
game_path = game_path_config.game_path

ROLE_ATTR_FILE_PATH = os.path.join(
    game_path, "data", language, "RoleAttributes.json"
)
role_attr_data = json_handle.load_json(ROLE_ATTR_FILE_PATH)
sex_data = role_attr_data["Sex"]
EQUIPMENT_FILE_PATH = os.path.join(
    game_path, "data", language, "Equipment.json"
)
equipment_data = json_handle.load_json(EQUIPMENT_FILE_PATH)


def get_sex_experience_text(sex_experience_data: dict, sex_name: str) -> list:
    """
    获取性经验描述文本
    Keyword arguments:
    sex_experience_data -- 性经验数据列表
    sex_name -- 性别
    """
    mouth_experience = (
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "19")
        + f"{sex_experience_data['mouth_experience']}"
    )
    bosom_experience = (
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "20")
        + f"{sex_experience_data['bosom_experience']}"
    )
    vagina_experience = (
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "21")
        + f"{sex_experience_data['vagina_experience']}"
    )
    clitoris_experience = (
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "22")
        + f"{sex_experience_data['clitoris_experience']}"
    )
    anus_experience = (
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "23")
        + f"{sex_experience_data['anus_experience']}"
    )
    penis_experience = (
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "24")
        + f"{sex_experience_data['penis_experience']}"
    )
    sex_experience_text = []
    sex_list = list(sex_data.keys())
    if sex_name == sex_list[0]:
        sex_experience_text = [
            mouth_experience,
            bosom_experience,
            anus_experience,
            penis_experience,
        ]
    elif sex_name == sex_list[1]:
        sex_experience_text = [
            mouth_experience,
            bosom_experience,
            vagina_experience,
            clitoris_experience,
            anus_experience,
        ]
    elif sex_name == sex_list[2]:
        sex_experience_text = [
            mouth_experience,
            bosom_experience,
            vagina_experience,
            clitoris_experience,
            anus_experience,
            penis_experience,
        ]
    elif sex_name == sex_list[3]:
        sex_experience_text = [
            mouth_experience,
            bosom_experience,
            anus_experience,
        ]
    return sex_experience_text


def get_sex_grade_text_list(sex_grade_data: dict, sex_name: str) -> list:
    """
    获取性等级描述文本
    Keyword arguments:
    sex_grade_data -- 性等级列表
    sex_name -- 性别
    """
    mouth_text = text_loading.get_text_data(
        constant.FilePath.STAGE_WORD_PATH, "25"
    ) + get_level_text_color(sex_grade_data["mouth_grade"])
    bosom_text = text_loading.get_text_data(
        constant.FilePath.STAGE_WORD_PATH, "26"
    ) + get_level_text_color(sex_grade_data["bosom_grade"])
    vagina_text = text_loading.get_text_data(
        constant.FilePath.STAGE_WORD_PATH, "27"
    ) + get_level_text_color(sex_grade_data["vagina_grade"])
    clitoris_text = text_loading.get_text_data(
        constant.FilePath.STAGE_WORD_PATH, "28"
    ) + get_level_text_color(sex_grade_data["clitoris_grade"])
    anus_text = text_loading.get_text_data(
        constant.FilePath.STAGE_WORD_PATH, "29"
    ) + get_level_text_color(sex_grade_data["anus_grade"])
    penis_text = text_loading.get_text_data(
        constant.FilePath.STAGE_WORD_PATH, "30"
    ) + get_level_text_color(sex_grade_data["penis_grade"])
    sex_grade_text_list = []
    sex_list = list(sex_data.keys())
    if sex_name == sex_list[0]:
        sex_grade_text_list = [mouth_text, bosom_text, anus_text, penis_text]
    elif sex_name == sex_list[1]:
        sex_grade_text_list = [
            mouth_text,
            bosom_text,
            vagina_text,
            clitoris_text,
            anus_text,
        ]
    elif sex_name == sex_list[2]:
        sex_grade_text_list = [
            mouth_text,
            bosom_text,
            vagina_text,
            clitoris_text,
            anus_text,
            penis_text,
        ]
    elif sex_name == sex_list[3]:
        sex_grade_text_list = [mouth_text, bosom_text, anus_text]
    return sex_grade_text_list


# 处理等级富文本
def get_level_text_color(level: str) -> str:
    """
    对等级文本进行富文本处理
    Keyword arguments:
    level -- 等级
    """
    lower_level = level.lower()
    return f"<level{lower_level}>{level}</level{lower_level}>"


def get_random_name_for_sex(sex_grade: str) -> str:
    """
    按性别随机生成姓名
    Keyword arguments:
    sex_grade -- 性别
    """
    family_random = random.randint(1, cache_contorl.family_region_int_list[-1])
    family_region_index = bisect.bisect_left(
        cache_contorl.family_region_int_list, family_random
    )
    family_region = cache_contorl.family_region_int_list[family_region_index]
    family_name = cache_contorl.family_region_list[family_region]
    if sex_grade == "Man":
        sex_judge = 1
    elif sex_grade == "Woman":
        sex_judge = 0
    else:
        sex_judge = random.randint(0, 1)
    if sex_judge == 0:
        name_random = random.randint(
            1, cache_contorl.girls_region_int_list[-1]
        )
        name_region_index = bisect.bisect_left(
            cache_contorl.girls_region_int_list, name_random
        )
        name_region = cache_contorl.girls_region_int_list[name_region_index]
        name = cache_contorl.girls_region_list[name_region]
    else:
        name_random = random.randint(1, cache_contorl.boys_region_int_list[-2])
        name_region_index = bisect.bisect_left(
            cache_contorl.boys_region_int_list, name_random
        )
        name_region = cache_contorl.boys_region_int_list[name_region_index]
        name = cache_contorl.boys_region_list[name_region]
    return family_name + name


def get_see_attr_panel_head_character_info(character_id: int) -> str:
    """
    获取查看角色属性面板头部角色缩略信息文本
    Keyword arguments:
    character_id -- 角色Id
    """
    character_data = cache_contorl.character_data["character"][character_id]
    character_id_text = (
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "0")
        + f"{character_id}"
    )
    name = character_data.name
    nick_name = character_data.nick_name
    character_name = (
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "13")
        + name
    )
    character_nick_name = (
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "12")
        + nick_name
    )
    sex = character_data.sex
    sex_text = text_loading.get_text_data(
        constant.FilePath.STAGE_WORD_PATH, "2"
    ) + get_sex_text(sex)
    name_text = (
        character_id_text
        + " "
        + character_name
        + " "
        + character_nick_name
        + " "
        + sex_text
    )
    return name_text


def get_sex_text(sex_id: str) -> str:
    """
    获取性别对应文本
    Keyword arguments:
    sex_id -- 性别
    """
    return role_attr_data["Sex"][sex_id]


def get_engraving_text(engraving_list: dict) -> list:
    """
    获取刻印描述文本
    Keyword arguments:
    e_list -- 刻印数据
    """
    pain_level = str(engraving_list["Pain"])
    happy_level = str(engraving_list["Happy"])
    yield_level = str(engraving_list["Yield"])
    fear_level = str(engraving_list["Fear"])
    resistance_level = str(engraving_list["Resistance"])
    pain_level_fix = text_loading.get_text_data(
        constant.FilePath.STAGE_WORD_PATH, "31"
    )
    happy_level_fix = text_loading.get_text_data(
        constant.FilePath.STAGE_WORD_PATH, "32"
    )
    yield_level_fix = text_loading.get_text_data(
        constant.FilePath.STAGE_WORD_PATH, "33"
    )
    fear_level_fix = text_loading.get_text_data(
        constant.FilePath.STAGE_WORD_PATH, "34"
    )
    resistance_level_fix = text_loading.get_text_data(
        constant.FilePath.STAGE_WORD_PATH, "35"
    )
    level_text = text_loading.get_text_data(
        constant.FilePath.STAGE_WORD_PATH, "36"
    )
    level_list = [
        pain_level,
        happy_level,
        yield_level,
        fear_level,
        resistance_level,
    ]
    level_fix_list = [
        pain_level_fix,
        happy_level_fix,
        yield_level_fix,
        fear_level_fix,
        resistance_level_fix,
    ]
    level_text_list = [
        f"{level_fix_list[i]}{level_text}{level_list[i]}"
        for i in range(len(level_list))
    ]
    level_bar_list = [
        f"{proportional_bar.get_count_bar(level_text_list[i],3,level_list[i],'engravingemptybar')}"
        for i in range(len(level_list))
    ]
    return level_bar_list


def get_clothing_text(clothing_list: dict) -> list:
    """
    获取服装描述文本
    Keyword arguments:
    clothing_list -- 服装数据
    """
    coat_id = int(clothing_list["Coat"])
    pants_id = int(clothing_list["Pants"])
    shoes_id = int(clothing_list["Shoes"])
    socks_id = int(clothing_list["Socks"])
    underwear_id = int(clothing_list["Underwear"])
    bra_id = int(clothing_list["Bra"])
    underpants_id = int(clothing_list["Underpants"])
    leggings_id = int(clothing_list["Leggings"])
    clothing_data = equipment_data["Clothing"]
    coat_text = clothing_data["Coat"][coat_id]
    pants_text = clothing_data["Pants"][pants_id]
    shoes_text = clothing_data["Shoes"][shoes_id]
    socks_text = clothing_data["Socks"][socks_id]
    underwear_text = clothing_data["Underwear"][underwear_id]
    bra_text = clothing_data["Bra"][bra_id]
    underpants_text = clothing_data["Underpants"][underpants_id]
    leggings_text = clothing_data["Leggings"][leggings_id]
    coat_text = (
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "41")
        + coat_text
    )
    pants_text = (
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "42")
        + pants_text
    )
    shoes_text = (
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "43")
        + shoes_text
    )
    socks_text = (
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "44")
        + socks_text
    )
    underwear_text = (
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "45")
        + underwear_text
    )
    bra_text = (
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "46")
        + bra_text
    )
    underpants_text = (
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "47")
        + underpants_text
    )
    leggings_text = (
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "48")
        + leggings_text
    )
    clothing_text_list = [
        coat_text,
        pants_text,
        shoes_text,
        socks_text,
        underwear_text,
        bra_text,
        underpants_text,
        leggings_text,
    ]
    return clothing_text_list


def get_gold_text(character_id: str) -> str:
    """
    获取指定角色的金钱信息描述文本
    Keyword arguments:
    character_id -- 角色id
    """
    gold_data = cache_contorl.character_data["character"][character_id].gold
    gold_data = str(gold_data)
    money_text = text_loading.get_text_data(
        constant.FilePath.STAGE_WORD_PATH, "66"
    )
    gold_text = text_loading.get_text_data(
        constant.FilePath.STAGE_WORD_PATH, "67"
    )
    gold_text = gold_text + gold_data + money_text
    return gold_text


def get_level_color_text(exp):
    """
    计算经验对应等级并获取富文本
    Keyword arguments:
    exp -- 经验
    """
    return get_level_text_color(attr_calculation.judge_grade(exp))


def get_state_text(character_id: str) -> str:
    """
    按角色Id获取状态描述信息
    Keyword arguments:
    character_id -- 角色Id
    """
    state = str(cache_contorl.character_data["character"][character_id].state)
    state_text = text_loading.get_text_data(
        constant.FilePath.STAGE_WORD_PATH, "132"
    )[state]
    return (
        text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "133")
        + state_text
    )


def judge_character_stature(character_id):
    """
    校验身材信息
    Keyword arguments:
    character_id -- 角色Id
    """
    character_data = cache_contorl.character_data["character"][character_id]
    target_sex = character_data.sex
    age_judge = "Similar"
    self_age = cache_contorl.character_data["character"][0].age
    target_age = character_data.age
    age_disparity = self_age - target_age
    if age_disparity < -2 and age_disparity >= -5:
        age_judge = "SlightlyHigh"
    elif age_disparity < -5 and age_disparity >= -15:
        age_judge = "High"
    elif age_disparity < -15 and age_disparity >= -30:
        age_judge = "VeryHigh"
    elif age_disparity < -30 and age_disparity >= -60:
        age_judge = "SuperHigh"
    elif age_disparity < -60:
        age_judge = "ExtremelyHigh"
    elif age_disparity > 2 and age_disparity <= 5:
        age_judge = "SlightlyLow"
    elif age_disparity > 5 and age_disparity <= 15:
        age_judge = "Low"
    elif age_disparity > 15 and age_disparity <= 30:
        age_judge = "VeryLow"
    elif age_disparity > 30 and age_disparity <= 60:
        age_judge = "SuperLow"
    elif age_disparity > 60:
        age_judge = "ExtremelyLow"
    bodyfat = character_data.bodyfat
    age_tem = attr_calculation.judge_age_group(target_age)
    average_bodyfat = cache_contorl.average_bodyfat_by_age[age_tem][target_sex]
    bodyfat_judge = "Similar"
    if bodyfat < average_bodyfat * 1.15 and bodyfat >= average_bodyfat * 1.05:
        bodyfat_judge = "SlilghtlyHeight"
    elif (
        bodyfat < average_bodyfat * 1.25 and bodyfat >= average_bodyfat * 1.15
    ):
        bodyfat_judge = "Height"
    elif (
        bodyfat < average_bodyfat * 1.35 and bodyfat >= average_bodyfat * 1.25
    ):
        bodyfat_judge = "VeryHeight"
    elif (
        bodyfat < average_bodyfat * 1.45 and bodyfat >= average_bodyfat * 1.35
    ):
        bodyfat_judge = "SuperHeight"
    elif bodyfat > average_bodyfat * 1.45:
        bodyfat_judge = "ExtremelyHeight"
    elif (
        bodyfat < average_bodyfat * 0.95 and bodyfat >= average_bodyfat * 0.85
    ):
        bodyfat_judge = "SlilghtlyLow"
    elif (
        bodyfat < average_bodyfat * 0.85 and bodyfat >= average_bodyfat * 0.75
    ):
        bodyfat_judge = "Low"
    elif (
        bodyfat < average_bodyfat * 0.75 and bodyfat >= average_bodyfat * 0.65
    ):
        bodyfat_judge = "VeryLow"
    elif (
        bodyfat < average_bodyfat * 0.65 and bodyfat >= average_bodyfat * 0.55
    ):
        bodyfat_judge = "SuperLow"
    elif bodyfat < average_bodyfat * 0.55:
        bodyfat_judge = "ExtremelyLow"
    average_height = cache_contorl.average_height_by_age[age_tem][target_sex]
    height = character_data.height["NowHeight"]
    height_judge = "Similar"
    if height < average_height * 1.15 and height >= average_height * 1.05:
        height_judge = "SlilghtlyHeight"
    elif height < average_height * 1.25 and height >= average_height * 1.15:
        height_judge = "Height"
    elif height < average_height * 1.35 and height >= average_height * 1.25:
        height_judge = "VeryHeight"
    elif height < average_height * 1.45 and height >= average_height * 1.35:
        height_judge = "SuperHeight"
    elif height > average_height * 1.45:
        height_judge = "ExtremelyHeight"
    elif height < average_height * 0.95 and height >= average_height * 0.85:
        height_judge = "SlilghtlyLow"
    elif height < average_height * 0.85 and height >= average_height * 0.75:
        height_judge = "Low"
    elif height < average_height * 0.75 and height >= average_height * 0.65:
        height_judge = "VeryLow"
    elif height < average_height * 0.65 and height >= average_height * 0.55:
        height_judge = "SuperLow"
    elif height < average_height:
        height_judge = "ExtremelyLow"
    player_bodyfat = character_data.bodyfat
    player_bodyfat_judge = "Similar"
    if bodyfat < player_bodyfat * 1.15 and bodyfat >= player_bodyfat * 1.05:
        player_bodyfat_judge = "SlilghtlyHeight"
    elif bodyfat < player_bodyfat * 1.25 and bodyfat >= player_bodyfat * 1.15:
        player_bodyfat_judge = "Height"
    elif bodyfat < player_bodyfat * 1.35 and bodyfat >= player_bodyfat * 1.25:
        player_bodyfat_judge = "VeryHeight"
    elif bodyfat < player_bodyfat * 1.45 and bodyfat >= player_bodyfat * 1.35:
        player_bodyfat_judge = "SuperHeight"
    elif bodyfat > player_bodyfat * 1.45:
        player_bodyfat_judge = "ExtremelyHeight"
    elif bodyfat < player_bodyfat * 0.95 and bodyfat >= player_bodyfat * 0.85:
        player_bodyfat_judge = "SlilghtlyLow"
    elif bodyfat < player_bodyfat * 0.85 and bodyfat >= player_bodyfat * 0.75:
        player_bodyfat_judge = "Low"
    elif bodyfat < player_bodyfat * 0.75 and bodyfat >= player_bodyfat * 0.65:
        player_bodyfat_judge = "VeryLow"
    elif bodyfat < player_bodyfat * 0.65 and bodyfat >= player_bodyfat * 0.55:
        player_bodyfat_judge = "SuperLow"
    elif bodyfat < player_bodyfat * 0.55:
        player_bodyfat_judge = "ExtremelyLow"
    player_height = cache_contorl.character_data["character"][0].height[
        "NowHeight"
    ]
    player_height_judge = "Similar"
    if height < player_height * 1.15 and height >= player_height * 1.05:
        player_height_judge = "SlilghtlyHeight"
    elif height < player_height * 1.25 and height >= player_height * 1.15:
        player_height_judge = "Height"
    elif height < player_height * 1.35 and height >= player_height * 1.25:
        player_height_judge = "VeryHeight"
    elif height < player_height * 1.45 and height >= player_height * 1.35:
        player_height_judge = "SuperHeight"
    elif height > player_height * 1.45:
        player_height_judge = "ExtremelyHeight"
    elif height < player_height * 0.95 and height >= player_height * 0.85:
        player_height_judge = "SlilghtlyLow"
    elif height < player_height * 0.85 and height >= player_height * 0.75:
        player_height_judge = "Low"
    elif height < player_height * 0.75 and height >= player_height * 0.65:
        player_height_judge = "VeryLow"
    elif height < player_height * 0.65 and height >= player_height * 0.55:
        player_height_judge = "SuperLow"
    elif height < player_height * 0.55:
        player_height_judge = "ExtremelyLow"
    player_sex = cache_contorl.character_data["character"][0].sex
    player_age = cache_contorl.character_data["character"][0].age
    player_age_tem = attr_calculation.judge_age_group(
        cache_contorl.character_data["character"][0].age
    )
    average_bodyfat = cache_contorl.average_bodyfat_by_age[player_age_tem][
        player_sex
    ]
    average_height = cache_contorl.average_height_by_age[player_age_tem][
        player_sex
    ]
    player_average_bodyfat_judge = "Similar"
    if (
        player_bodyfat < average_bodyfat * 1.15
        and player_bodyfat >= average_bodyfat * 1.05
    ):
        player_average_bodyfat_judge = "SlilghtlyHeight"
    elif (
        player_bodyfat < average_bodyfat * 1.25
        and player_bodyfat >= average_bodyfat * 1.15
    ):
        player_average_bodyfat_judge = "Height"
    elif (
        player_bodyfat < average_bodyfat * 1.35
        and player_bodyfat >= average_bodyfat * 1.25
    ):
        player_average_bodyfat_judge = "VeryHeight"
    elif (
        player_bodyfat < average_bodyfat * 1.45
        and player_bodyfat >= average_bodyfat * 1.35
    ):
        player_average_bodyfat_judge = "SuperHeight"
    elif player_bodyfat > average_bodyfat * 1.45:
        player_average_bodyfat_judge = "ExtremelyHeight"
    elif (
        player_bodyfat < average_bodyfat * 0.95
        and player_bodyfat >= average_bodyfat * 0.85
    ):
        player_average_bodyfat_judge = "SlilghtlyLow"
    elif (
        player_bodyfat < average_bodyfat * 0.85
        and player_bodyfat >= average_bodyfat * 0.75
    ):
        player_average_bodyfat_judge = "Low"
    elif (
        player_bodyfat < average_bodyfat * 0.75
        and player_bodyfat >= average_bodyfat * 0.65
    ):
        player_average_bodyfat_judge = "VeryLow"
    elif (
        player_bodyfat < average_bodyfat * 0.65
        and player_bodyfat >= average_bodyfat * 0.55
    ):
        player_average_bodyfat_judge = "SuperLow"
    elif player_bodyfat < average_bodyfat * 0.55:
        player_average_bodyfat_judge = "ExtremelyLow"
    player_average_height_judge = "Similar"
    if (
        player_height < average_height * 1.15
        and player_height >= average_height * 1.05
    ):
        player_average_height_judge = "SlilghtlyHeight"
    elif (
        player_height < average_height * 1.25
        and player_height >= average_height * 1.15
    ):
        player_average_height_judge = "Height"
    elif (
        player_height < average_height * 1.35
        and player_height >= average_height * 1.25
    ):
        player_average_height_judge = "VeryHeight"
    elif (
        player_height < average_height * 1.45
        and player_height >= average_height * 1.35
    ):
        player_average_height_judge = "SuperHeight"
    elif player_height > average_height * 1.45:
        player_average_height_judge = "ExtremelyHeight"
    elif (
        player_height < average_height * 0.95
        and player_height >= average_height * 0.85
    ):
        player_average_height_judge = "SlilghtlyLow"
    elif (
        player_height < average_height * 0.85
        and player_height >= average_height * 0.75
    ):
        player_average_height_judge = "Low"
    elif (
        player_height < average_height * 0.75
        and player_height >= average_height * 0.65
    ):
        player_average_height_judge = "VeryLow"
    elif (
        player_height < average_height * 0.65
        and player_height >= average_height * 0.55
    ):
        player_average_height_judge = "SuperLow"
    elif player_height < average_height * 0.55:
        player_average_height_judge = "ExtremelyLow"
    target_judge = "Target"
    if character_id == 0:
        target_judge = "Self"
    return {
        "SelfSex": player_sex,
        "TargetSex": character_data.sex,
        "AgeJudge": age_judge,
        "AgeTem": age_tem,
        "SelfAgeTem": player_age_tem,
        "SelfAge": player_age,
        "TargetAge": target_age,
        "AverageHeight": height_judge,
        "AverageStature": bodyfat_judge,
        "RelativeHeight": player_height_judge,
        "RelativeStature": player_bodyfat_judge,
        "PlayerAverageHeight": player_average_height_judge,
        "PlayerAverageStature": player_average_bodyfat_judge,
        "Target": target_judge,
    }


def get_stature_text(character_id: str) -> list:
    """
    按角色Id获取身材描述信息
    Keyword arguments:
    character_id -- 角色Id
    """
    stature_judge = judge_character_stature(character_id)
    for priority in cache_contorl.stature_descrition_priorition_data:
        for descript in cache_contorl.stature_descrition_priorition_data[
            priority
        ]:
            if judge_stature_description(
                stature_judge,
                text_loading.get_game_data(
                    constant.FilePath.STATURE_DESCRIPTION_PATH
                )["Priority"][priority][descript]["Condition"],
            ):
                return text_loading.get_game_data(
                    constant.FilePath.STATURE_DESCRIPTION_PATH
                )["Priority"][priority][descript]["Description"]
    return ""


def judge_stature_description(stature_judge, description_data):
    """
    角色的身材信息
    Keyword arguments:
    stature_judge -- 身材校验信息
    description_data -- 身材描述数据
    """
    for judge in description_data:
        if (
            judge == "SelfAge"
            and stature_judge["SelfAge"] < description_data[judge]
        ):
            return False
        elif (
            judge == "TargetAge"
            and stature_judge["TargetAge"] < description_data[judge]
        ):
            return False
        else:
            if description_data[judge] != stature_judge[judge]:
                return False
    return True


def get_character_abbreviations_info(character_id: int) -> str:
    """
    按角色id获取角色缩略信息文本
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache_contorl.character_data["character"][character_id]
    character_id_info = text_loading.get_text_data(
        constant.FilePath.STAGE_WORD_PATH, "0"
    )
    character_id_text = f"{character_id_info}{character_id}"
    character_name = character_data.name
    character_sex = character_data.sex
    character_sex_info = text_loading.get_text_data(
        constant.FilePath.STAGE_WORD_PATH, "2"
    )
    character_sex_text_data = text_loading.get_text_data(
        constant.FilePath.ROLE_PATH, "Sex"
    )
    character_sex_text = character_sex_text_data[character_sex]
    character_sex_text = character_sex_info + character_sex_text
    character_age = character_data.age
    character_age_info = text_loading.get_text_data(
        constant.FilePath.STAGE_WORD_PATH, "3"
    )
    character_age_text = character_age_info + str(character_age)
    character_hp_and_mp_text = attr_print.get_hp_and_mp_text(character_id)
    character_intimate = character_data.intimate
    character_intimate_info = text_loading.get_text_data(
        constant.FilePath.STAGE_WORD_PATH, "16"
    )
    character_intimate_text = character_intimate_info + str(character_intimate)
    character_graces = character_data.graces
    character_graces_info = text_loading.get_text_data(
        constant.FilePath.STAGE_WORD_PATH, "17"
    )
    character_graces_text = character_graces_info + str(character_graces)
    abbreviations_info = (
        character_id_text
        + " "
        + character_name
        + " "
        + character_sex_text
        + " "
        + character_age_text
        + " "
        + character_hp_and_mp_text
        + " "
        + character_intimate_text
        + " "
        + character_graces_text
    )
    return abbreviations_info


def get_character_dormitory_path_text(character_id: str) -> str:
    """
    获取角色宿舍路径描述信息
    Keyword arguments:
    character_id -- 角色Id
    Return arguments:
    map_path_str -- 宿舍路径描述文本
    """
    dormitory = cache_contorl.character_data["character"][
        character_id
    ].dormitory
    dormitory_path = map_handle.get_map_system_path_for_str(dormitory)
    map_list = map_handle.get_map_hierarchy_list_for_scene_path(
        dormitory_path, []
    )
    map_path_text = text_loading.get_text_data(
        constant.FilePath.STAGE_WORD_PATH, "143"
    )
    map_list.reverse()
    for now_map in map_list:
        now_map_map_system_str = map_handle.get_map_system_path_str_for_list(
            now_map
        )
        map_name = cache_contorl.map_data[now_map_map_system_str]["MapName"]
        map_path_text += map_name + "-"
    map_path_text += cache_contorl.scene_data[dormitory]["SceneName"]
    return map_path_text
