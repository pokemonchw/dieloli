import os
import random
from Script.Core import (
    cache_contorl,
    game_config,
    game_path_config,
    text_loading,
    json_handle,
    value_handle,
    constant,
)
from Script.Design import game_time

language = game_config.language
game_path = game_path_config.game_path
role_attr_path = os.path.join(
    game_path, "data", language, "RoleAttributes.json"
)
role_attr_data = json_handle.load_json(role_attr_path)


def get_tem_list() -> dict:
    """
    获取人物生成模板
    """
    return text_loading.get_text_data(
        constant.FilePath.ATTR_TEMPLATE_PATH, "TemList"
    )


def get_features_list() -> dict:
    """
    获取特征模板
    """
    return role_attr_data["Features"]


def get_age_tem_list() -> list:
    """
    获取年龄模板
    """
    return list(
        text_loading.get_text_data(
            constant.FilePath.ATTR_TEMPLATE_PATH, "AgeTem"
        ).keys()
    )


def get_engraving_list() -> dict:
    """
    获取刻印列表
    """
    return role_attr_data["Default"]["Engraving"]


def get_gold() -> int:
    """
    获取默认金钱数据
    """
    return role_attr_data["Default"]["Gold"]


def get_age(tem_name: str) -> int:
    """
    按年龄模板id随机生成年龄数据
    Keyword arguments:
    tem_name -- 年龄模板id
    """
    tem_data = text_loading.get_text_data(
        constant.FilePath.ATTR_TEMPLATE_PATH, "AgeTem"
    )[tem_name]
    max_age = int(tem_data["MaxAge"])
    mini_age = int(tem_data["MiniAge"])
    return random.randint(mini_age, max_age)


def get_end_age(sex: str) -> int:
    """
    按照性别模板随机生成预期寿命
    Keyword arguments:
    sex -- 性别
    """
    tem_value = text_loading.get_text_data(
        constant.FilePath.ATTR_TEMPLATE_PATH, "EndAgeTem"
    )[sex]
    return random.randint(int(tem_value * 0.5), int(tem_value * 1.5))


def get_height(tem_name: str, age: int) -> dict:
    """
    按模板和年龄计算身高
    Keyword arguments:
    tem_name -- 人物生成模板
    age -- 人物年龄
    """
    tem_data = text_loading.get_text_data(
        constant.FilePath.ATTR_TEMPLATE_PATH, "HeightTem"
    )[tem_name]
    initial_height = random.uniform(tem_data[0], tem_data[1])
    if tem_name == "Man" or "Asexual":
        expect_age = random.randint(18, 22)
        expect_height = initial_height / 0.2949
    else:
        expect_age = random.randint(13, 17)
        expect_height = initial_height / 0.3109
    development_age = random.randint(4, 6)
    growth_height_data = get_growth_height(
        age, expect_height, development_age, expect_age
    )
    growth_height = growth_height_data["GrowthHeight"]
    now_height = growth_height_data["NowHeight"]
    if age > expect_age:
        now_height = expect_height
    else:
        now_height = 365 * growth_height * age + now_height
    return {
        "NowHeight": now_height,
        "GrowthHeight": growth_height,
        "ExpectAge": expect_age,
        "DevelopmentAge": development_age,
        "ExpectHeight": expect_height,
    }


def get_chest(chest_tem: str, birthday: dict):
    """
    按罩杯模板生成人物最终罩杯，并按人物年龄计算当前罩杯
    Keyword arguments:
    chest_tem -- 罩杯模板
    birthday -- 出生日期
    """
    target_chest = get_rand_npc_chest(chest_tem)
    over_age = random.randint(14, 18)
    over_year = birthday["year"] + over_age
    end_date = game_time.get_rand_day_for_year(over_year).timetuple()
    now_date = cache_contorl.game_time.copy()
    now_date = game_time.game_time_to_time_tuple(now_date)
    start_date = game_time.game_time_to_time_tuple(birthday)
    end_day = game_time.count_day_for_time_tuple(start_date, end_date)
    now_day = game_time.count_day_for_time_tuple(start_date, now_date)
    sub_chest = target_chest / end_day
    now_chest = sub_chest * now_day
    if now_chest > sub_chest:
        now_chest = target_chest
    return {
        "TargetChest": target_chest,
        "NowChest": now_chest,
        "SubChest": sub_chest,
    }


chest_tem_weight_data = text_loading.get_text_data(
    constant.FilePath.ATTR_TEMPLATE_PATH, "ChestWeightTem"
)


def get_rand_npc_chest_tem() -> str:
    """
    随机获取npc罩杯模板
    """
    return value_handle.get_random_for_weight(chest_tem_weight_data)


def get_rand_npc_chest(chest_tem: str) -> int:
    """
    随机获取模板对应罩杯
    Keyword arguments:
    chest_tem -- 罩杯模板
    """
    chest_scope = text_loading.get_text_data(
        constant.FilePath.ATTR_TEMPLATE_PATH, "ChestTem"
    )[chest_tem]
    return random.uniform(chest_scope[0], chest_scope[1])


def get_rand_npc_birthday(age: int):
    """
    随机生成npc生日
    Keyword arguments:
    age -- 年龄
    """
    now_year = int(cache_contorl.game_time["year"])
    now_month = int(cache_contorl.game_time["month"])
    now_day = int(cache_contorl.game_time["day"])
    birth_year = now_year - age
    date = game_time.get_rand_day_for_year(birth_year)
    birthday = {
        "year": date.year,
        "month": date.month,
        "day": date.day,
        "hour": date.hour,
        "minute": date.minute,
    }
    if now_month < birthday["month"] or (
        now_month == birthday["month"] and now_day < birthday["day"]
    ):
        birthday["year"] -= 1
    return birthday


def get_growth_height(
    now_age: int, expect_height: float, development_age: int, expect_age: int
) -> dict:
    """
    计算每日身高增长量
    Keyword arguments:
    now_age -- 现在的年龄
    expect_height -- 预期最终身高
    development_age -- 结束发育期时的年龄
    except_age -- 结束身高增长时的年龄
    """
    if now_age > development_age:
        now_height = expect_height / 2
        judge_age = expect_age - development_age
        growth_height = now_height / (judge_age * 365)
    else:
        judge_height = expect_height / 2
        now_height = 0
        growth_height = judge_height / (now_age * 365)
    return {"GrowthHeight": growth_height, "NowHeight": now_height}


def get_bmi(tem_name: str) -> dict:
    """
    按体重比例模板生成BMI
    Keyword arguments:
    tem_name -- 体重比例模板id
    """
    tem_data = text_loading.get_text_data(
        constant.FilePath.ATTR_TEMPLATE_PATH, "WeightTem"
    )[tem_name]
    return random.uniform(tem_data[0], tem_data[1])


def get_bodyfat(sex: str, tem_name: str) -> float:
    """
    按性别和体脂率模板生成体脂率
    Keyword arguments:
    sex -- 性别
    tem_name -- 体脂率模板id
    """
    if sex in ["Man", "Asexual"]:
        sex_tem = "Man"
    else:
        sex_tem = "Woman"
    tem_data = text_loading.get_text_data(
        constant.FilePath.ATTR_TEMPLATE_PATH, "BodyFatTem"
    )[sex_tem][tem_name]
    return random.uniform(tem_data[0], tem_data[1])


def get_weight(bmi: float, height: float) -> float:
    """
    按bmi和身高计算体重
    Keyword arguments:
    bmi -- 身高体重比(BMI)
    height -- 身高
    """
    height = height / 100
    return bmi * height * height


def get_measurements(
    tem_name: str,
    height: float,
    weight: float,
    bodyfat: float,
    weight_tem: str,
) -> dict:
    """
    计算角色三围
    Keyword arguments:
    tem_name -- 性别模板
    height -- 身高
    weight -- 体重
    bodyfat -- 体脂率
    weight_tem -- 体重比例模板
    """
    if tem_name == "Man" or "Asexual":
        bust = 51.76 / 100 * height
        waist = 42.79 / 100 * height
        hip = 52.07 / 100 * height
        new_waist = (
            (bodyfat / 100 * weight) + (weight * 0.082 + 34.89)
        ) / 0.74
    else:
        bust = 52.35 / 100 * height
        waist = 41.34 / 100 * height
        hip = 57.78 / 100 * height
        new_waist = (
            (bodyfat / 100 * weight) + (weight * 0.082 + 44.74)
        ) / 0.74
    waist_hip_proportion = waist / hip
    waist_hip_proportion_tem = text_loading.get_text_data(
        constant.FilePath.ATTR_TEMPLATE_PATH, "WaistHipProportionTem"
    )[weight_tem]
    waist_hip_proportion_fix = random.uniform(0, waist_hip_proportion_tem)
    waist_hip_proportion = waist_hip_proportion + waist_hip_proportion_fix
    new_hip = new_waist / waist_hip_proportion
    fix = new_hip / hip
    bust = bust * fix
    return {"Bust": bust, "Waist": new_waist, "Hip": new_hip}


def get_max_hit_point(tem_name: str) -> int:
    """
    获取最大hp值
    Keyword arguments:
    tem_name -- hp模板
    """
    tem_data = text_loading.get_text_data(
        constant.FilePath.ATTR_TEMPLATE_PATH, "HitPointTem"
    )[tem_name]
    max_hit_point = int(tem_data["HitPointMax"])
    add_value = random.randint(0, 500)
    impairment = random.randint(0, 500)
    return max_hit_point + add_value - impairment


def get_max_mana_point(tem_name: str) -> int:
    """
    获取最大mp值
    Keyword arguments:
    tem_name -- mp模板
    """
    tem_data = text_loading.get_text_data(
        constant.FilePath.ATTR_TEMPLATE_PATH, "ManaPointTem"
    )[tem_name]
    max_mana_point = int(tem_data["ManaPointMax"])
    add_value = random.randint(0, 500)
    impairment = random.randint(0, 500)
    return max_mana_point + add_value - impairment


def get_init_learn_abllity(age: int, end_age: int):
    """
    按年龄和生成学习能力
    Keyword arguments:
    age -- 年龄
    end_age -- 预期寿命
    """
    stop_age = int(end_age / 2)
    forget_age = int(end_age * 0.9)
    if age <= stop_age:
        return age / stop_age
    elif age > stop_age and age < end_age:
        return 1 - 1 / (forget_age - stop_age) * (forget_age - age)
    else:
        return 0 - (age - forget_age) / (end_age - forget_age)


def get_sex_experience(tem_name: str) -> dict:
    """
    按模板生成角色初始性经验
    Keyword arguments:
    tem_name -- 性经验模板
    """
    tem_data = text_loading.get_text_data(
        constant.FilePath.ATTR_TEMPLATE_PATH, "SexExperience"
    )[tem_name]
    mouth_experience_tem_name = tem_data["MouthExperienceTem"]
    bosom_experience_tem_name = tem_data["BosomExperienceTem"]
    vagina_experience_tem_name = tem_data["VaginaExperienceTem"]
    clitoris_experience_tem_name = tem_data["ClitorisExperienceTem"]
    anus_experience_tem_name = tem_data["AnusExperienceTem"]
    penis_experience_tem_name = tem_data["PenisExperienceTem"]
    mouth_experience_list = text_loading.get_text_data(
        constant.FilePath.ATTR_TEMPLATE_PATH, "SexExperienceTem"
    )["MouthExperienceTem"][mouth_experience_tem_name]
    mouth_experience = random.randint(
        int(mouth_experience_list[0]), int(mouth_experience_list[1])
    )
    bosom_experience_list = text_loading.get_text_data(
        constant.FilePath.ATTR_TEMPLATE_PATH, "SexExperienceTem"
    )["BosomExperienceTem"][bosom_experience_tem_name]
    bosom_experience = random.randint(
        int(bosom_experience_list[0]), int(bosom_experience_list[1])
    )
    vagina_experience_list = text_loading.get_text_data(
        constant.FilePath.ATTR_TEMPLATE_PATH, "SexExperienceTem"
    )["VaginaExperienceTem"][vagina_experience_tem_name]
    vagina_experience = random.randint(
        int(vagina_experience_list[0]), int(vagina_experience_list[1])
    )
    clitoris_experience_list = text_loading.get_text_data(
        constant.FilePath.ATTR_TEMPLATE_PATH, "SexExperienceTem"
    )["ClitorisExperienceTem"][clitoris_experience_tem_name]
    clitoris_experience = random.randint(
        int(clitoris_experience_list[0]), int(clitoris_experience_list[1])
    )
    anus_experience_list = text_loading.get_text_data(
        constant.FilePath.ATTR_TEMPLATE_PATH, "SexExperienceTem"
    )["AnusExperienceTem"][anus_experience_tem_name]
    anus_experience = random.randint(
        int(anus_experience_list[0]), int(anus_experience_list[1])
    )
    penis_experience_list = text_loading.get_text_data(
        constant.FilePath.ATTR_TEMPLATE_PATH, "SexExperienceTem"
    )["PenisExperienceTem"][penis_experience_tem_name]
    penis_experience = random.randint(
        int(penis_experience_list[0]), int(penis_experience_list[1])
    )
    return {
        "mouth_experience": mouth_experience,
        "bosom_experience": bosom_experience,
        "vagina_experience": vagina_experience,
        "clitoris_experience": clitoris_experience,
        "anus_experience": anus_experience,
        "penis_experience": penis_experience,
    }


def get_sex_grade(sex_experience_data: dict) -> dict:
    """
    按性经验数据计算性经验等级
    Keyword arguments:
    sex_experience_data -- 性经验数据
    """
    mouth_experience = sex_experience_data["mouth_experience"]
    bosom_experience = sex_experience_data["bosom_experience"]
    vagina_experience = sex_experience_data["vagina_experience"]
    clitoris_experience = sex_experience_data["clitoris_experience"]
    anus_experience = sex_experience_data["anus_experience"]
    penis_experience = sex_experience_data["penis_experience"]
    mouth_grade = judge_grade(mouth_experience)
    bosom_grade = judge_grade(bosom_experience)
    vagina_grade = judge_grade(vagina_experience)
    clitoris_grade = judge_grade(clitoris_experience)
    anus_grade = judge_grade(anus_experience)
    penis_grade = judge_grade(penis_experience)
    return {
        "mouth_grade": mouth_grade,
        "bosom_grade": bosom_grade,
        "vagina_grade": vagina_grade,
        "clitoris_grade": clitoris_grade,
        "anus_grade": anus_grade,
        "penis_grade": penis_grade,
    }


def judge_grade(experience: int) -> float:
    """
    按经验数值评定等级
    Keyword arguments:
    experience -- 经验数值
    """
    grade = ""
    if experience < 50:
        grade = "G"
    elif experience < 100:
        grade = "F"
    elif experience < 200:
        grade = "E"
    elif experience < 500:
        grade = "D"
    elif experience < 1000:
        grade = "C"
    elif experience < 2000:
        grade = "B"
    elif experience < 5000:
        grade = "A"
    elif experience < 10000:
        grade = "S"
    elif experience >= 10000:
        grade = "EX"
    return grade


def judge_age_group(age: int):
    """
    判断所属年龄段
    Keyword arguments:
    age -- 年龄
    """
    age_group = text_loading.get_game_data(
        constant.FilePath.ATTR_TEMPLATE_PATH
    )["AgeTem"]
    for age_tem in age_group:
        if int(age) >= int(age_group[age_tem]["MiniAge"]) and int(age) < int(
            age_group[age_tem]["MaxAge"]
        ):
            return age_tem
    return "YoundAdult"


def judge_chest_group(chest: int):
    """
    判断胸围差所属罩杯
    Keyword arguments:
    chest -- 胸围差
    """
    chest_group = text_loading.get_game_data(
        constant.FilePath.ATTR_TEMPLATE_PATH
    )["ChestTem"]
    for chest_tem in chest_group:
        if (
            int(chest) >= int(chest_group[chest_tem][0])
            and int(chest) < chest_group[chest_tem][1]
        ):
            return chest_tem
