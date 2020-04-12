import random
import math
import uuid
import time
import itertools
import numpy
from Script.Core import (
    cache_contorl,
    value_handle,
    game_data,
    text_loading,
    game_path_config,
    game_config,
)
from Script.Design import (
    attr_calculation,
    map_handle,
    attr_text,
    clothing,
    nature,
    character,
)

language = game_config.language
game_path = game_path_config.game_path
sex_list = list(text_loading.get_text_data(text_loading.ROLE_PATH, "Sex"))
age_tem_list = list(
    text_loading.get_text_data(text_loading.ATTR_TEMPLATE_PATH, "AgeTem")
)
character_list = list(game_data.game_data[language]["character"].keys())


def init_character_list():
    """
    初始生成所有npc数据
    """
    init_character_tem()
    id_list = iter([i + 1 for i in range(len(cache_contorl.npc_tem_data))])
    npc_data_iter = iter(cache_contorl.npc_tem_data)
    while 1:
        try:
            init_character(next(id_list), next(npc_data_iter))
        except StopIteration:
            break
    index_character_average_value()
    calculate_the_average_value_of_each_attribute_of_each_age_group()


def calculate_the_average_value_of_each_attribute_of_each_age_group():
    """
    计算各年龄段各项属性平均值
    """
    cache_contorl.average_bodyfat_by_age = {
        sex: {
            age_tem: cache_contorl.total_bodyfat_by_age[sex][age_tem]
            / cache_contorl.total_number_of_people_of_all_ages[sex][age_tem]
            for age_tem in cache_contorl.total_bodyfat_by_age[sex]
        }
        for sex in cache_contorl.total_bodyfat_by_age
    }
    cache_contorl.average_height_by_age = {
        sex: {
            age_tem: cache_contorl.total_height_by_age[sex][age_tem]
            / cache_contorl.total_number_of_people_of_all_ages[sex][age_tem]
            for age_tem in cache_contorl.total_height_by_age[sex]
        }
        for sex in cache_contorl.total_height_by_age
    }


def index_character_average_value():
    """
    统计各年龄段所有角色各属性总值
    """
    for character in cache_contorl.character_data["character"]:
        character_data = cache_contorl.character_data["character"][character]
        age_tem = attr_calculation.judge_age_group(character_data.age)
        cache_contorl.total_height_by_age.setdefault(age_tem, {})
        cache_contorl.total_height_by_age[age_tem].setdefault(character_data.sex, 0)
        cache_contorl.total_height_by_age[age_tem][
            character_data.sex
        ] += character_data.height["NowHeight"]
        cache_contorl.total_number_of_people_of_all_ages.setdefault(age_tem, {})
        cache_contorl.total_number_of_people_of_all_ages[age_tem].setdefault(
            character_data.sex, 0
        )
        cache_contorl.total_number_of_people_of_all_ages[age_tem][
            character_data.sex
        ] += 1
        cache_contorl.total_bodyfat_by_age.setdefault(age_tem, {})
        cache_contorl.total_bodyfat_by_age[age_tem].setdefault(character_data.sex, 0)
        cache_contorl.total_bodyfat_by_age[age_tem][
            character_data.sex
        ] += character_data.bodyfat


def init_character(character_id: int, character_tem: dict):
    """
    按id生成角色属性
    Keyword arguments:
    character_id -- 角色id
    character_tem -- 角色生成模板数据
    """
    now_character = character.Character()
    now_character.name = character_tem["Name"]
    now_character.sex = character_tem["Sex"]
    if "MotherTongue" in character_tem:
        now_character.mother_tongue = character_tem["MotherTongue"]
    if "Age" in character_tem:
        now_character.age = attr_calculation.get_age(character_tem["Age"])
    if "Weight" in character_tem:
        now_character.weigt_tem = character_tem["Weight"]
    if "BodyFat" in character_tem:
        now_character.bodyfat_tem = character_tem["BodyFat"]
    else:
        now_character.bodyfat_tem = now_character.weigt_tem
    if "Chest" in character_tem:
        now_character.chest_tem = character_tem["Chest"]
    now_character.init_attr()
    cache_contorl.character_data["character"][character_id] = now_character


def init_character_tem():
    """
    初始化角色模板数据
    """
    init_random_npc_data()
    npc_data = cache_contorl.random_npc_list
    now_characterList = character_list.copy()
    npc_data += [get_dir_character_tem(character) for character in now_characterList]
    numpy.random.shuffle(npc_data)
    cache_contorl.npc_tem_data = npc_data


def get_dir_character_tem(character: int) -> dict:
    """
    获取预设角色模板数据
    """
    return text_loading.get_character_data(character)["AttrTemplate"]


random_npc_max = int(game_config.random_npc_max)
random_teacher_proportion = int(game_config.proportion_teacher)
random_student_proportion = int(game_config.proportion_student)
age_weight_data = {
    "Teacher": random_teacher_proportion,
    "Student": random_student_proportion,
}
age_weight_regin_data = value_handle.get_region_list(age_weight_data)
age_weight_regin_list = list(map(int, age_weight_regin_data.keys()))
age_weight_max = sum(
    [int(age_weight_data[age_weight]) for age_weight in age_weight_data]
)


def init_random_npc_data() -> list:
    """
    生成所有随机npc的数据模板
    """
    cache_contorl.random_npc_list = []
    for i in range(random_npc_max):
        create_random_npc(i)


def create_random_npc(id) -> dict:
    """
    生成随机npc数据模板
    """
    now_age_weight = random.randint(-1, age_weight_max - 1)
    now_age_weight_regin = value_handle.get_next_value_for_list(
        now_age_weight, age_weight_regin_list
    )
    age_weight_tem = age_weight_regin_data[now_age_weight_regin]
    random_npc_sex = get_rand_npc_sex()
    random_npc_name = attr_text.get_random_name_for_sex(random_npc_sex)
    random_npc_age_tem = get_rand_npc_age_tem(age_weight_tem)
    fat_tem = get_rand_npc_fat_tem(age_weight_tem)
    body_fat_tem = get_rand_npc_body_fat_tem(age_weight_tem, fat_tem)
    random_npc_new_data = {
        "Name": random_npc_name,
        "Sex": random_npc_sex,
        "Age": random_npc_age_tem,
        "Position": ["0"],
        "AdvNpc": "1",
        "Weight": fat_tem,
        "BodyFat": body_fat_tem,
    }
    if random_npc_sex in {"Woman": 1, "Futa": 1}:
        random_npc_new_data["Chest"] = attr_calculation.get_rand_npc_chest_tem()
    else:
        random_npc_new_data["Chest"] = "Precipice"
    cache_contorl.random_npc_list.append(random_npc_new_data)


sex_weight_data = text_loading.get_text_data(
    text_loading.ATTR_TEMPLATE_PATH, "RandomNpcSexWeight"
)
sex_weight_max = sum([int(sex_weight_data[weight]) for weight in sex_weight_data])
sex_weight_regin_data = value_handle.get_region_list(sex_weight_data)
sex_weight_regin_list = list(map(int, sex_weight_regin_data.keys()))


def get_rand_npc_sex() -> str:
    """
    随机获取npc性别
    """
    now_weight = random.randint(0, sex_weight_max - 1)
    weight_regin = value_handle.get_next_value_for_list(
        now_weight, sex_weight_regin_list
    )
    return sex_weight_regin_data[weight_regin]


fat_weight_data = text_loading.get_text_data(
    text_loading.ATTR_TEMPLATE_PATH, "FatWeight"
)


def get_rand_npc_fat_tem(age_judge: str) -> str:
    """
    按人群年龄段体重分布比例随机生成体重模板
    Keyword arguments:
    agejudge -- 年龄段
    """
    now_fat_weight_data = fat_weight_data[age_judge]
    now_fat_tem = value_handle.get_random_for_weight(now_fat_weight_data)
    return now_fat_tem


def get_rand_npc_sexExperienceTem(age: int, sex: str) -> str:
    """
    按年龄范围随机获取性经验模板
    Keyword arguments:
    age -- 年龄
    sex -- 性别
    """
    age_judge_sex_experience_tem_data = text_loading.get_text_data(
        text_loading.ATTR_TEMPLATE_PATH, "AgeJudgeSexExperienceTem"
    )
    if sex == "Asexual":
        sex = "Woman"
    if sex == "Futa":
        sex = "Man"
    now_tem_data = age_judge_sex_experience_tem_data[sex]
    age_region_list = [int(i) for i in now_tem_data.keys()]
    age_region = str(value_handle.get_old_value_for_list(age, age_region_list))
    age_regionData = now_tem_data[age_region]
    return value_handle.get_random_for_weight(age_regionData)


body_fat_weight_data = text_loading.get_text_data(
    text_loading.ATTR_TEMPLATE_PATH, "BodyFatWeight"
)


def get_rand_npc_body_fat_tem(age_judge: str, bmi_tem: str) -> str:
    """
    按年龄段体脂率分布比例随机生成体脂率模板
    Keyword arguments:
    age_judge -- 年龄段
    bmi_tem -- bmi模板
    """
    now_body_fat_data = body_fat_weight_data[age_judge][bmi_tem]
    return value_handle.get_random_for_weight(now_body_fat_data)


age_tem_weight_data = text_loading.get_text_data(
    text_loading.ATTR_TEMPLATE_PATH, "AgeWeight"
)


def get_rand_npc_age_tem(age_judge: str) -> int:
    """
    按年龄断随机生成npc年龄
    Keyword arguments:
    age_judge -- 年龄段
    """
    now_age_weight_data = age_tem_weight_data[age_judge]
    now_age_tem = value_handle.get_random_for_weight(now_age_weight_data)
    return now_age_tem


def init_character_dormitory():
    """
    分配角色宿舍
    小于18岁，男生分配到男生宿舍，女生分配到女生宿舍，按宿舍楼层和角色年龄，从下往上，从小到大分配，其他性别分配到地下室，大于18岁，教师宿舍混居
    """
    character_data = {}
    character_sex_data = {
        "Man": {
            character_id: cache_contorl.character_data["character"][character_id].age
            for character_id in cache_contorl.character_data["character"]
            if cache_contorl.character_data["character"][character_id].age < 18
            and cache_contorl.character_data["character"][character_id].sex == "Man"
        },
        "Woman": {
            character_id: cache_contorl.character_data["character"][character_id].age
            for character_id in cache_contorl.character_data["character"]
            if cache_contorl.character_data["character"][character_id].age < 18
            and cache_contorl.character_data["character"][character_id].sex == "Woman"
        },
        "Other": {
            character_id: cache_contorl.character_data["character"][character_id].age
            for character_id in cache_contorl.character_data["character"]
            if cache_contorl.character_data["character"][character_id].age < 18
            and cache_contorl.character_data["character"][character_id].sex
            not in {"Man": 0, "Woman": 1}
        },
        "Teacher": {
            character_id: cache_contorl.character_data["character"][character_id].age
            for character_id in cache_contorl.character_data["character"]
            if cache_contorl.character_data["character"][character_id].age >= 18
        },
    }
    man_max = len(character_sex_data["Man"])
    woman_max = len(character_sex_data["Woman"])
    other_max = len(character_sex_data["Other"])
    teacher_max = len(character_sex_data["Teacher"])
    character_sex_data["Man"] = [
        k[0] for k in sorted(character_sex_data["Man"].items(), key=lambda x: x[1])
    ]
    character_sex_data["Woman"] = [
        k[0] for k in sorted(character_sex_data["Woman"].items(), key=lambda x: x[1])
    ]
    character_sex_data["Other"] = [
        k[0] for k in sorted(character_sex_data["Other"].items(), key=lambda x: x[1])
    ]
    character_sex_data["Teacher"] = [
        k[0] for k in sorted(character_sex_data["Teacher"].items(), key=lambda x: x[1])
    ]
    teacher_dormitory = {
        x: 0
        for x in sorted(
            cache_contorl.place_data["TeacherDormitory"], key=lambda x: x[0]
        )
    }
    male_dormitory = {
        key: cache_contorl.place_data[key]
        for key in cache_contorl.place_data
        if "MaleDormitory" in key
    }
    female_dormitory = {
        key: cache_contorl.place_data[key]
        for key in cache_contorl.place_data
        if "FemaleDormitory" in key
    }
    male_dormitory = {
        x: 0
        for j in [k[1] for k in sorted(male_dormitory.items(), key=lambda x: x[0])]
        for x in j
    }
    female_dormitory = {
        x: 0
        for j in [k[1] for k in sorted(female_dormitory.items(), key=lambda x: x[0])]
        for x in j
    }
    basement = {x: 0 for x in cache_contorl.place_data["Basement"]}
    male_dormitoryMax = len(male_dormitory.keys())
    female_dormitoryMax = len(female_dormitory.keys())
    teacher_dormitoryMax = len(teacher_dormitory)
    basement_max = len(basement)
    single_room_man = math.ceil(man_max / male_dormitoryMax)
    single_room_woman = math.ceil(woman_max / female_dormitoryMax)
    single_room_basement = math.ceil(other_max / basement_max)
    single_room_teacher = math.ceil(teacher_max / teacher_dormitoryMax)
    man_list = character_sex_data["Man"].copy()
    for character in character_sex_data["Man"]:
        now_room = list(male_dormitory.keys())[0]
        cache_contorl.character_data["character"][character].dormitory = now_room
        male_dormitory[now_room] += 1
        if male_dormitory[now_room] >= single_room_man:
            del male_dormitory[now_room]
    for character in character_sex_data["Woman"]:
        now_room = list(female_dormitory.keys())[0]
        cache_contorl.character_data["character"][character].dormitory = now_room
        female_dormitory[now_room] += 1
        if female_dormitory[now_room] >= single_room_woman:
            del female_dormitory[now_room]
    for character in character_sex_data["Other"]:
        now_room = list(basement.keys())[0]
        cache_contorl.character_data["character"][character].dormitory = now_room
        basement[now_room] += 1
        if basement[now_room] >= single_room_basement:
            del basement[now_room]
    for character in character_sex_data["Teacher"]:
        now_room = list(teacher_dormitory.keys())[0]
        cache_contorl.character_data["character"][character].dormitory = now_room
        teacher_dormitory[now_room] += 1
        if teacher_dormitory[now_room] >= single_room_teacher:
            del teacher_dormitory[now_room]


def init_character_position():
    """
    初始化角色位置
    """
    for character in cache_contorl.character_data["character"]:
        character_position = cache_contorl.character_data["character"][
            character
        ].position
        character_dormitory = cache_contorl.character_data["character"][
            character
        ].dormitory
        character_dormitory = map_handle.get_map_system_path_for_str(
            character_dormitory
        )
        map_handle.character_move_scene(
            character_position, character_dormitory, character
        )
