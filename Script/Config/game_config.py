import os
import configparser
from typing import Dict,List
from Script.Config import config_def
from Script.Core import game_type,json_handle,get_text


data_path = os.path.join("data","data.json")
""" 原始json数据文件路径 """
config_data = {}
""" 原始json数据 """
config_age_judge_sex_experience_tem:Dict[int,config_def.AgeJudgeSexExperienceTem] = {}
""" 不同性别不同年龄段对应生成不同性经验模板的权重 """
config_age_judge_sex_experience_tem_data:Dict[int,Dict[int,Dict[int,int]]] = {}
"""
不同性别不同年龄段对应生成不同性经验模板的配置数据
性别:年龄段:性经验模板:权重
"""
config_age_tem:Dict[int,config_def.AgeTem] = {}
""" 年龄段年龄范围模板 """
config_attr_tem:Dict[int,config_def.AttrTem] = {}
""" 性别对应角色各项基础属性模板 """
config_bar:Dict[int,config_def.BarConfig] = {}
""" 比例条配置数据 """
config_bar_data:Dict[str,int] = {}
""" 比例条名字对应比例条id """
config_body_fat_tem:Dict[int,config_def.BodyFatTem] = {}
""" 按性别划分的体脂率模板和范围 """
config_body_fat_tem_data:Dict[int,Dict[int,int]] = {}
"""
性别划分的体脂率模板对应范围数据
性别:体脂率范围:体脂率范围配置id
"""
config_book:Dict[int,config_def.Book] = {}
""" 书籍配表数据 """
config_character_statue:Dict[int,config_def.CharacterState] = {}
""" 角色状态属性配表数据 """
config_character_statue_type:Dict[int,config_def.CharacterStateType] = {}
""" 角色状态类型配表数据 """
config_chest:Dict[int,config_def.ChestTem] = {}
""" 罩杯配置数据 """
config_clothing_collocational:Dict[int,config_def.ClothingCollocational] = {}
""" 服装搭配模板配置 """
config_clothing_collocational_data:Dict[int,Dict[int,List[config_def.ClothingCollocational]]] = {}
"""
服装搭配模板配置数据
服装id:搭配位置:搭配配置列表
"""
config_clothing_evaluate:Dict[int,config_def.ClothingEvaluate] = {}
""" 服装评价配置数据 """
config_clothing_evaluate_list:List[str] = []
""" 服装评价配置列表 """
config_clothing_suit:Dict[int,config_def.ClothingSuit] = {}
""" 衣服套装配置列表 """
config_clothing_suit_data:Dict[int,Dict[int,set]] = {}
"""
衣服套装搭配数据
套装编号:性别id:服装集合
"""
config_clothing_tag:Dict[int,config_def.ClothingTag] = {}
""" 服装属性标签配置数据 """
config_clothing_tem:Dict[int,config_def.ClothingTem] = {}
""" 服装模板配置数据 """
config_clothing_type:Dict[int,config_def.ClothingType] = {}
""" 衣服种类配置数据 """
config_clothing_use_type:Dict[int,config_def.ClothingUseType] = {}
""" 衣服用途配置数据 """
config_end_age_tem:Dict[int,config_def.EndAgeTem] = {}
""" 最终年龄范围配置模板 """
config_end_age_tem_sex_data:Dict[int,int] = {}
""" 各性别最终年龄范围配置数据 """
config_font:Dict[int,config_def.FontConfig] = {}
""" 字体配置数据 """
config_font_data:Dict[str,int] = {}
""" 字体名字对应字体id """
config_food:Dict[int,config_def.Food] = {}
""" 食材配置数据 """
config_food_feel:Dict[int,config_def.FoodFeel] = {}
""" 食材效果配置 """
config_food_feel_data:Dict[int,Dict[int,int]] = {}
"""
食材效果配置数据
食材id:效果id:效果
"""
config_food_quality_weight:Dict[int,config_def.FoodQualityWeight] = {}
""" 烹饪技能等级制造食物品质权重配置 """
config_food_quality_weight_data:Dict[int,Dict[int,int]] = {}
"""
烹饪技能等级制造食物品质权重表
技能等级:食物品质:权重
"""
config_height_tem:Dict[int,config_def.HeightTem] = {}
""" 身高预期值模板 """
config_height_tem_sex_data:Dict[int,config_def.HeightTem] = {}
""" 性别对应身高预期值模板 """
config_hitpoint_tem:Dict[int,config_def.HitPointTem] = {}
""" HP模板对应平均值 """
config_manapoint_tem:Dict[int,config_def.ManaPointTem] = {}
""" MP模板对应平均值 """
config_nature:Dict[int,config_def.Nature] = {}
""" 性格配置数据 """
config_nature_tag:Dict[int,config_def.NatureTag] = {}
""" 性格标签配置数据 """
config_organ:Dict[int,config_def.Organ] = {}
""" 器官种类配置 """
config_random_sex_weight:Dict[int,config_def.RandomNpcSexWeight] = {}
""" 生成随机npc时性别权重配置数据 """
config_recipes:Dict[int,config_def.Recipes] = {}
""" 菜谱配置 """
config_recipes_formula:Dict[int,config_def.RecipesFormula] = {}
""" 菜谱配方配置 """
config_recipes_formula_type:Dict[int,config_def.RecipesFormulaType] = {}
""" 菜谱配方材料类型配置 """
config_sex_experience:Dict[int,config_def.SexExperience] = {}
""" 性经验丰富程度模板对应器官性经验模板 """
config_sex_experience_tem:Dict[int,config_def.SexExperienceTem] = {}
""" 器官类型性经验丰富程度对应经验范围 """
config_sex_tem:Dict[int,config_def.SexTem] = {}
""" 性别对应描述和性别器官模板 """
config_week_day:Dict[int,config_def.WeekDay] = {}
""" 星期描述文本配置数据 """
config_weight_tem:Dict[int,config_def.WeightTem] = {}
""" 体重模板对应体重范围 """


def load_data_json():
    """ 载入data.json内配置数据 """
    global config_data
    config_data = json_handle.load_json(data_path)



def translate_data(data:dict):
    """
    按指定字段翻译数据
    Keyword arguments:
    data -- 待翻译的字典数据
    """
    if "get_text" not in data:
        return
    for key in data["data"]:
        if data["get_text"][key]:
            data[key] = get_text._(data[key])


def load_age_judge_sex_experience_tem_data():
    """ 载入不同性别不同年龄段对应生成不同性经验模板的权重 """
    now_data = config_data["AgeJudgeSexExperienceTem"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_tem = config_def.AgeJudgeSexExperienceTem()
        now_tem.__dict__ = tem_data
        config_age_judge_sex_experience_tem[now_tem.cid] = now_tem
        config_age_judge_sex_experience_tem_data.setdefault(now_tem.sex,{})
        config_age_judge_sex_experience_tem_data[now_tem.sex].setdefault(now_tem.age,{})
        config_age_judge_sex_experience_tem_data[now_tem.sex][now_tem.age][now_tem.sex_exp_tem] = now_tem.weight


def load_age_tem():
    """ 载入各年龄段对应年龄范围模板 """
    now_data = config_data["AgeTem"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_tem = config_def.AgeTem()
        now_tem.__dict__ = tem_data
        config_age_tem[now_tem.cid] = now_tem


def load_attr_tem():
    """ 载入性别对应角色各项基础属性模板 """
    now_data = config_data["AttrTem"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_tem = config_def.AttrTem()
        now_tem.__dict__ = tem_data
        config_attr_tem[now_tem.cid] = now_tem


def load_bar_data():
    """ 载入比例条配置数据 """
    now_data = config_data["BarConfig"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_bar = config_def.BarConfig()
        now_bar.__dict__ = tem_data
        config_bar[now_bar.cid] = now_bar
        config_bar_data[now_bar.name] = now_bar.cid


def load_body_fat_tem():
    """ 载入按性别划分的体脂率模板和范围配置数据 """
    now_data = config_data["BodyFatTem"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_tem = config_def.BodyFatTem()
        now_tem.__dict__ = tem_data
        config_body_fat_tem[now_tem.cid] = now_tem
        config_body_fat_tem_data.setdefault(now_tem.sex_type,{})
        config_body_fat_tem_data[now_tem.sex_type][now_tem.sub_type] = now_tem.cid


def load_book_data():
    """ 载入数据配置数据 """
    now_data = config_data["Book"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_tem = config_def.Book()
        now_tem.__dict__ = tem_data
        config_book[now_tem.cid] = now_tem


def load_character_statue_data():
    """ 载入角色状态属性配表数据 """
    now_data = config_data["CharacterState"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_tem = config_def.CharacterState()
        now_tem.__dict__ = tem_data
        config_character_statue[now_tem.cid] = now_tem


def load_character_statue_type_data():
    """ 载入角色状态类型配表数据 """
    now_data = config_data["CharacterStateType"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_tem = config_def.CharacterStateType()
        now_tem.__dict__ = tem_data
        config_character_statue_type[now_tem.cid] = now_tem


def load_chest_tem_data():
    """ 载入罩杯配置数据 """
    now_data = config_data["ChestTem"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_chest = config_def.ChestTem()
        now_chest.__dict__ = tem_data
        config_chest[now_chest.cid] = now_chest


def load_clothing_collocational():
    """ 载入服装搭配模板配置数据 """
    now_data = config_data["ClothingCollocational"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_collocational = config_def.ClothingCollocational()
        now_collocational.__dict__ = tem_data
        config_clothing_collocational[now_collocational.cid] = now_collocational
        config_clothing_collocational_data.setdefault(now_collocational.clothing_tem_type,{})
        config_clothing_collocational_data[now_collocational.clothing_tem_type].setdefault(now_collocational.clothing_tem,[])
        config_clothing_collocational_data[now_collocational.clothing_tem_type][now_collocational.clothing_tem].append(now_collocational)


def load_clothing_evaluate():
    """ 载入服装评价配置数据 """
    now_data = config_data["ClothingEvaluate"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_tem = config_def.ClothingEvaluate()
        now_tem.__dict__ = tem_data
        config_clothing_evaluate[now_tem.cid] = now_tem
        config_clothing_evaluate_list.append(now_tem.name)


def load_clothing_suit():
    """ 载入衣服套装配置数据 """
    now_data = config_data["ClothingSuit"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_tem = config_def.ClothingSuit()
        now_tem.__dict__ = tem_data
        config_clothing_suit[now_tem.cid] = now_tem
        config_clothing_suit_data.setdefault(now_tem.suit_type,{})
        config_clothing_suit_data[now_tem.suit_type].setdefault(now_tem.sex,set())
        config_clothing_suit_data[now_tem.suit_type][now_tem.sex].add(now_tem.clothing_id)


def load_clothing_tag():
    """ 载入服装属性标签配置数据 """
    now_data = config_data["ClothingTag"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_tem = config_def.ClothingTag()
        now_tem.__dict__ = tem_data
        config_clothing_tag[now_tem.cid] = now_tem


def load_clothing_tem():
    """ 载入服装模板配置数据 """
    now_data = config_data["ClothingTem"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_tem = config_def.ClothingTem()
        now_tem.__dict__ = tem_data
        config_clothing_tem[now_tem.cid] = now_tem


def load_clothing_type():
    """ 载入衣服种类配置数据 """
    now_data = config_data["ClothingType"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_type = config_def.ClothingType()
        now_type.__dict__ = tem_data
        config_clothing_type[now_type.cid] = now_type


def load_clothing_use_type():
    """ 载入衣服用途配置数据 """
    now_data = config_data["ClothingUseType"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_type = config_def.ClothingUseType()
        now_type.__dict__ = tem_data
        config_clothing_use_type[now_type.cid] = now_type


def load_end_age_tem():
    """ 载入最终年龄范围配置模板 """
    now_data = config_data["EndAgeTem"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_tem = config_def.EndAgeTem()
        now_tem.__dict__ = tem_data
        config_end_age_tem[now_tem.cid] = now_tem
        config_end_age_tem_sex_data[now_tem.sex] = now_tem.end_age


def load_font_data():
    """ 载入字体配置数据 """
    now_data = config_data["FontConfig"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_font = config_def.FontConfig()
        now_font.__dict__ = tem_data
        config_font[now_font.cid] = now_font
        config_font_data[now_font.name] = now_font.cid


def load_food_data():
    """ 载入食材配置数据 """
    now_data = config_data["Food"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_tem = config_def.Food()
        now_tem.__dict__ = tem_data
        config_food[now_tem.cid] = now_tem


def load_food_feel_data():
    """ 载入食材效果配置数据 """
    now_data = config_data["FoodFeel"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_tem = config_def.FoodFeel()
        now_tem.__dict__ = tem_data
        config_food_feel[now_tem.cid] = now_tem
        config_food_feel_data.setdefault(now_tem.food_id,{})
        config_food_feel_data[now_tem.food_id][now_tem.feel_id] = now_tem.feel_value


def load_food_quality_weight():
    """ 载入烹饪技能等级制造食物品质权重配置数据 """
    now_data = config_data["FoodQualityWeight"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_tem = config_def.FoodQualityWeight()
        now_tem.__dict__ = tem_data
        config_food_quality_weight[now_tem.cid] = now_tem
        config_food_quality_weight_data.setdefault(now_tem.level,{})
        config_food_quality_weight_data[now_tem.level][now_tem.quality] = now_tem.weight


def load_height_tem():
    """ 载入身高预期值模板 """
    now_data = config_data["HeightTem"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_tem = config_def.HeightTem()
        now_tem.__dict__ = tem_data
        config_height_tem[now_tem.cid] = now_tem
        config_height_tem_sex_data[now_tem.sex] = now_tem


def load_hitpoint_tem():
    """ 载入hp模板对应平均值配置数据 """
    now_data = config_data["HitPointTem"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_tem = config_def.HitPointTem()
        now_tem.__dict__ = tem_data
        config_hitpoint_tem[now_tem.cid] = now_tem


def load_manapoint_tem():
    """ 载入mp模板对应平均值配置数据 """
    now_data = config_data["ManaPointTem"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_tem = config_def.ManaPointTem()
        now_tem.__dict__ = tem_data
        config_manapoint_tem[now_tem.cid] = now_tem


def load_nature():
    """ 载入性格配置数据 """
    now_data = config_data["Nature"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_tem = config_def.Nature()
        now_tem.__dict__ = tem_data
        config_nature[now_tem.cid] = now_tem


def load_nature_tag():
    """ 载入性格标签配置数据 """
    now_data = config_data["NatureTag"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_tem = config_def.NatureTag()
        now_tem.__dict__ = tem_data
        config_nature_tag[now_tem.cid] = now_tem


def load_organ_data():
    """ 载入器官种类配置 """
    now_data = config_data["Organ"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_tem = config_def.Organ()
        now_tem.__dict__ = tem_data
        config_organ[now_tem.cid] = now_tem


def load_random_sex_weight():
    """ 载入生成随机npc时性别权重配置数据 """
    now_data = config_data["RandomNpcSexWeight"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_tem = config_def.RandomNpcSexWeight()
        now_tem.__dict__ = tem_data
        config_random_sex_weight[now_tem.cid] = now_tem


def load_recipes():
    """ 载入菜谱配置数据 """
    now_data = config_data["Recipes"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_tem = config_def.Recipes()
        now_tem.__dict__ = tem_data
        config_recipes[now_tem.cid] = now_tem


def load_recipes_formula():
    """ 载入菜谱配方配置 """
    now_data = config_data["RecipesFormula"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_tem = config_def.RecipesFormula()
        now_tem.__dict__ = tem_data
        config_recipes_formula[now_tem.cid] = now_tem


def load_recipes_formula_type():
    """ 载入菜谱配方材料类型配置数据 """
    now_data = config_data["RecipesFormulaType"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_tem = config_def.RecipesFormulaType()
        now_tem.__dict__ = tem_data
        config_recipes_formula_type[now_tem.cid] = now_tem


def load_sex_experience():
    """ 载入性经验丰富模板对应器官性经验模板配置数据 """
    now_data = config_data["SexExperience"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_tem = config_def.SexExperience()
        now_tem.__dict__ = tem_data
        config_sex_experience[now_tem.cid] = now_tem


def load_sex_experience_tem():
    """ 载入器官类型性经验丰富程度对应经验范围 """
    now_data = config_data["SexExperienceTem"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_tem = config_def.SexExperienceTem()
        now_tem.__dict__ = tem_data
        config_sex_experience_tem[now_tem.cid] = now_tem


def load_sex_tem():
    """ 载入性别对应描述和性别器官模板数据 """
    now_data = config_data["SexTem"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_tem = config_def.SexTem()
        now_tem.__dict__ = tem_data
        config_sex_tem[now_tem.cid] = now_tem


def load_week_day():
    """ 载入星期描述文本配置数据 """
    now_data = config_data["WeekDay"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_tem = config_def.WeekDay()
        now_tem.__dict__ = tem_data
        config_week_day[now_tem.cid] = now_tem


def load_weight_tem():
    """ 载入体重模板对应体重范围 """
    now_data = config_data["WeightTem"]
    translate_data(now_data)
    for tem_data in now_data["data"]:
        now_tem = config_def.WeightTem()
        now_tem.__dict__ = tem_data
        config_weight_tem[now_tem.cid] = now_tem


def init():
    """ 初始化游戏配置数据 """
    load_data_json()
    init_normal_config()
    load_age_judge_sex_experience_tem_data()
    load_age_tem()
    load_attr_tem()
    load_bar_data()
    load_body_fat_tem()
    load_book_data()
    load_character_statue_data()
    load_character_statue_type_data()
    load_chest_tem_data()
    load_clothing_collocational()
    load_clothing_evaluate()
    load_clothing_suit()
    load_clothing_tag()
    load_clothing_tem()
    load_clothing_type()
    load_clothing_use_type()
    load_end_age_tem()
    load_font_data()
    load_food_data()
    load_food_feel_data()
    load_food_quality_weight()
    load_height_tem()
    load_hitpoint_tem()
    load_manapoint_tem()
    load_nature()
    load_nature_tag()
    load_organ_data()
    load_random_sex_weight()
    load_recipes()
    load_recipes_formula()
    load_recipes_formula_type()
    load_sex_experience()
    load_sex_experience_tem()
    load_sex_tem()
    load_week_day()
    load_weight_tem()
