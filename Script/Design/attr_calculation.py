import random
import datetime
import numpy
from Script.Core import (
    cache_control,
    value_handle,
    game_type,
)
from Script.Design import game_time
from Script.Config import game_config

cache: game_type.Cache = cache_control.cache
""" 游戏内缓存数据 """


def get_age_tem_list() -> list:
    """
    获取年龄模板
    """
    return list(game_config.config_age_tem.keys())


def get_age(tem_name: int) -> int:
    """
    按年龄模板id随机生成年龄数据
    Keyword arguments:
    tem_name -- 年龄模板id
    """
    tem_data = game_config.config_age_tem[tem_name]
    max_age = tem_data.max_age
    mini_age = tem_data.min_age
    return random.randint(mini_age, max_age)


def get_end_age(sex: int) -> int:
    """
    按照性别模板随机生成预期寿命
    Keyword arguments:
    sex -- 性别
    """
    tem_value = game_config.config_end_age_tem_sex_data[sex]
    return value_handle.get_beta_rand(int(tem_value * 0.5), int(tem_value * 1.5))


def get_height(tem_name: int, age: int) -> game_type.Height:
    """
    按模板和年龄计算身高
    Keyword arguments:
    tem_name -- 人物生成模板(性别id)
    age -- 人物年龄
    """
    tem_data = game_config.config_height_tem_sex_data[tem_name]
    initial_height = value_handle.get_beta_rand(tem_data.min_value, tem_data.max_value)
    expect_age = 0
    expect_height = 0
    if tem_name in {0, 3}:
        expect_age = random.randint(16,20)
        expect_height = initial_height / 0.2949
    else:
        expect_age = random.randint(13, 17)
        expect_height = initial_height / 0.3109
    current_height, daily_growth = predict_height(initial_height, expect_height, age, expect_age,tem_name)
    height_data = game_type.Height()
    height_data.now_height = current_height
    height_data.growth_height = daily_growth
    height_data.expect_age = expect_age
    height_data.expect_height = expect_height
    height_data.birth_height = initial_height
    return height_data


def get_chest(chest_tem: int, birthday: int) -> game_type.Chest:
    """
    按罩杯模板生成人物最终胸围差，并按人物年龄计算当前胸围差
    Keyword arguments:
    chest_tem -- 罩杯模板
    birthday -- 出生日期
    Return arguments:
    game_type.Chest -- 胸围数据
    """
    target_chest = get_rand_npc_chest(chest_tem)
    over_age = int(value_handle.get_beta_rand(14, 18))
    over_year_start = birthday + (over_age - 1) * 31536365
    over_year_end = birthday + over_age * 31536365
    end_date = random.randint(over_year_start, over_year_end)
    now_date = cache.game_time
    end_day = int((end_date - birthday) / 86400)
    now_day = int((now_date - birthday) / 86400)
    sub_chest = target_chest / end_day
    now_chest = sub_chest * now_day
    now_chest = min(now_chest, target_chest)
    chest = game_type.Chest()
    chest.now_chest = now_chest
    chest.sub_chest = sub_chest
    chest.target_chest = target_chest
    return chest


chest_tem_weight_data = {k: game_config.config_chest[k].weight for k in game_config.config_chest}


def get_rand_npc_chest_tem() -> int:
    """
    随机获取npc罩杯模板
    """
    return value_handle.get_random_for_weight(chest_tem_weight_data)


def get_rand_npc_chest(chest_tem: int) -> int:
    """
    随机获取模板对应罩杯
    Keyword arguments:
    chest_tem -- 罩杯模板
    """
    chest_scope = game_config.config_chest[chest_tem]
    return value_handle.get_beta_rand(chest_scope.min_value, chest_scope.max_value)


def get_rand_npc_birthday(age: int) -> int:
    """
    随机生成npc生日
    Keyword arguments:
    age -- 年龄
    Return arguments:
    int -- 生日
    """
    now_date = datetime.datetime.fromtimestamp(cache.game_time, game_time.time_zone)
    now_year = now_date.year
    now_month = now_date.month
    now_day = now_date.day
    birth_year = now_year - age
    birthday = game_time.get_rand_day_for_year(birth_year)
    if birthday > 0:
        birthday_data = datetime.datetime.fromtimestamp(birthday, game_time.time_zone)
    else:
        birthday_data = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=birthday)
    if now_month < birthday_data.month or (
            now_month == birthday_data.month and now_day < birthday_data.day
    ):
        birthday = game_time.get_sub_date(year=-1, old_date=birthday)
    return birthday


def logistic_growth(t: float, k: float, r: float, t0: float) -> float:
    """
    逻辑生长函数，用于模拟身高增长。

    :param t: 当前年龄。
    :param k: 生长的最大值。
    :param r: 生长速率。
    :param t0: 生长曲线的中点时间。
    :return: 在年龄 t 时的身高。
    """
    return k / (1 + numpy.exp(-r * (t - t0)))

def segmented_growth(age: float, birth_height: float, k: float, r1: float, t1: float, r2: float, t2: float, r3: float, t3: float) -> float:
    """
    分段生长函数，用于模拟不同生长阶段的身高增长。
    Keyword arguments:
    age -- 当前年龄
    k -- 预期身高
    r1 -- 婴儿时期生长速率
    t1 -- 婴儿时期结束时间
    r2 -- 青春期生长速率
    t2 -- 青春期结束时间
    r3 -- 青春期结束后生长速率
    t3 -- 结束生长时间
    Return arguments:
    float -- 当前身高
    """
    if age < t1:
        return birth_height + logistic_growth(age, k/2 - birth_height, r1, t1/2)
    elif age < t2:
        return birth_height + logistic_growth(t1, k/2 - birth_height, r1, t1/2) + logistic_growth(age - t1, k/4, r2, (t2-t1)/2)
    elif age < t3:
        return birth_height + logistic_growth(t1, k/2 - birth_height, r1, t1/2) + logistic_growth(t2-t1, k/4, r2, (t2-t1)/2) + logistic_growth(age - t2, k/4, r3, (t3-t2)/2)
    else:
        return k

def predict_height(birth_height: float, final_height: float, current_age: float, age_at_max_height: float, gender: int) -> (float, float):
    """
    预测当前身高和每日增长率。
    Keyword arguments:
    birth_height -- 当前身高
    final_height -- 预期身高
    current_age -- 当前年龄
    age_at_max_height -- 达到最大身高的年龄
    gender -- 性别
    Return arguments:
    float -- 当前身高
    float -- 每日身高增量
    """
    k = final_height  # 最大身高
    # 根据性别调整婴幼儿期和青春期的参数
    if gender in {0, 3}:
        r1, t1 = 0.8, 2  # 男性婴幼儿期
        t2_start, t2_end = 12, age_at_max_height  # 男性青春期
    else:
        r1, t1 = 0.9, 2  # 女性婴幼儿期
        t2_start, t2_end = 10, age_at_max_height - 1  # 女性青春期
    r2, t2 = 0.4, (t2_start + t2_end) / 2
    r3 = 1.2 / (t2_end - t2_start)
    current_height = segmented_growth(current_age, birth_height, k, r1, t1, r2, t2, r3, t2_end)
    # 计算每日增长率
    delta = 0.001
    future_height = segmented_growth(current_age + delta, birth_height, k, r1, t1, r2, t2, r3, t2_end)
    daily_growth = (future_height - current_height) / delta
    return current_height, daily_growth / 365.25


def get_bmi(tem_name: int) -> float:
    """
    按体重比例模板生成BMI
    Keyword arguments:
    tem_name -- 体重比例模板id
    Return arguments:
    int -- bmi值
    """
    tem_data = game_config.config_weight_tem[tem_name]
    return value_handle.get_beta_rand(tem_data.min_value, tem_data.max_value)


def get_body_fat(sex: int, tem_name: int) -> float:
    """
    按性别和体脂率模板生成体脂率
    Keyword arguments:
    sex -- 性别
    tem_name -- 体脂率模板id
    Return arguments:
    float -- 体脂率
    """
    sex_tem = sex in (0, 3)
    tem_data_id = game_config.config_body_fat_tem_data[sex_tem][tem_name]
    tem_data = game_config.config_body_fat_tem[tem_data_id]
    return value_handle.get_beta_rand(tem_data.min_value, tem_data.max_value)


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
        tem_name: int,
        height: float,
        weight_tem: int,
) -> game_type.Measurements:
    """
    计算角色三围
    Keyword arguments:
    tem_name -- 性别模板
    height -- 身高
    weight_tem -- 体重比例模板
    """
    fix = 0
    if not weight_tem:
        fix = -5
    elif weight_tem > 1:
        fix = 5 * (weight_tem - 1)
    if tem_name in {0, 3}:
        bust = value_handle.get_beta_rand(0.4676, 0.5676) * height + fix
        waist = value_handle.get_beta_rand(0.3779, 0.4779) * height + fix
        hip = value_handle.get_beta_rand(0.4707, 0.5707) * height + fix
    else:
        bust = value_handle.get_beta_rand(0.4735, 0.5735) * height + fix
        waist = value_handle.get_beta_rand(0.3634, 0.4634) * height + fix
        hip = value_handle.get_beta_rand(0.5278, 0.6278) * height + fix
    measurements = game_type.Measurements()
    measurements.bust = bust
    measurements.waist = waist
    measurements.hip = hip
    return measurements


def get_max_hit_point(tem_id: int) -> int:
    """
    获取最大hp值
    Keyword arguments:
    tem_id -- hp模板id
    Return arguments:
    int -- 最大hp值
    """
    tem_data = game_config.config_hitpoint_tem[tem_id]
    max_hit_point = tem_data.max_value
    add_value = value_handle.get_beta_rand(0, 500)
    impairment = value_handle.get_beta_rand(0, 500)
    return max_hit_point + add_value - impairment


def get_max_mana_point(tem_id: int) -> int:
    """
    获取最大mp值
    Keyword arguments:
    tem_id -- mp模板
    Return arguments:
    int -- 最大mp值
    """
    tem_data = game_config.config_manapoint_tem[tem_id]
    max_mana_point = tem_data.max_value
    add_value = value_handle.get_beta_rand(0, 500)
    impairment = value_handle.get_beta_rand(0, 500)
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
    if stop_age < age < end_age:
        return 1 - 1 / (forget_age - stop_age) * (forget_age - age)
    return 0 - (age - forget_age) / (end_age - forget_age)


def get_sex_experience(tem_name: int, sex: int) -> dict:
    """
    按模板生成角色初始性经验
    Keyword arguments:
    tem_name -- 性经验丰富程度模板
    sex -- 性别id
    Return arguments:
    dict -- 性经验数据
    """
    sex_tem = sex in {0, 3}
    sex_experience_data = {}
    for organ in game_config.config_organ_data[sex_tem] | game_config.config_organ_data[2]:
        sex_experience_tem_id = game_config.config_sex_experience_data[tem_name][organ]
        organ_sex_experience_tem_id = game_config.config_sex_experience_tem_data[organ][
            sex_experience_tem_id
        ]
        organ_sex_experience_tem = game_config.config_sex_experience_tem[
            organ_sex_experience_tem_id
        ]
        sex_experience_data[organ] = random.uniform(
            organ_sex_experience_tem.min_exp, organ_sex_experience_tem.max_exp
        )
    return sex_experience_data


def get_experience_level_weight(experience: int) -> int:
    """
    按经验计算技能等级权重
    Keyword arguments:
    experience -- 经验数值
    Return arguments:
    int -- 权重
    """
    grade = 0
    if experience < 100:
        grade = 0
    elif experience < 500:
        grade = 1
    elif experience < 1000:
        grade = 2
    elif experience < 2000:
        grade = 3
    elif experience < 3000:
        grade = 4
    elif experience < 5000:
        grade = 5
    elif experience < 10000:
        grade = 6
    elif experience < 20000:
        grade = 7
    elif experience >= 20000:
        grade = 8
    return grade


def judge_grade(experience: int) -> str:
    """
    按经验数值评定等级
    Keyword arguments:
    experience -- 经验数值
    Return arguments:
    str -- 评级
    """
    grade = ""
    if experience < 100:
        grade = "G"
    elif experience < 500:
        grade = "F"
    elif experience < 1000:
        grade = "E"
    elif experience < 2000:
        grade = "D"
    elif experience < 3000:
        grade = "C"
    elif experience < 5000:
        grade = "B"
    elif experience < 10000:
        grade = "A"
    elif experience < 20000:
        grade = "S"
    elif experience >= 20000:
        grade = "EX"
    return grade


def judge_age_group(age: int) -> int:
    """
    判断所属年龄段
    Keyword arguments:
    age -- 年龄
    Return arguments:
    int -- 年龄段
    """
    for age_tem_id in game_config.config_age_tem:
        age_tem = game_config.config_age_tem[age_tem_id]
        if age_tem.min_age <= age < age_tem.max_age:
            return age_tem_id
    return 0


def judge_chest_group(chest: float) -> int:
    """
    判断胸围差所属罩杯
    Keyword arguments:
    chest -- 胸围差
    Return arguments:
    int -- 罩杯id
    """
    for chest_tem_id in game_config.config_chest:
        chest_tem = game_config.config_chest[chest_tem_id]
        if chest_tem.min_value <= chest < chest_tem.max_value:
            return chest_tem_id
    return 0
