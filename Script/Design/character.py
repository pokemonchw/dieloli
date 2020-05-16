import random
import uuid
from Script.Core import cache_contorl, text_loading, value_handle, constant
from Script.Design import attr_calculation, clothing, nature


class Character:
    def __init__(self):
        self.name = "主人公"
        self.nick_name = "你"
        self.self_name = "我"
        self.species = "人类"
        self.sex = "Man"
        self.age = 17
        self.end_age = 74
        self.relationship = "无"
        self.intimate = 0
        self.graces = 0
        self.features = {}
        self.hit_point_max = 0
        self.hit_point = 0
        self.mana_point_max = 0
        self.mana_point = 0
        self.learning_ability = (1,)
        self.sex_experience = {}
        self.sex_grade = {}
        self.state = 0
        self.engraving = {}
        self.clothing = {
            "Coat": {},
            "Underwear": {},
            "Pants": {},
            "Skirt": {},
            "Shoes": {},
            "Socks": {},
            "Bra": {},
            "Underpants": {},
        }
        self.sex_item = {}
        self.item = {}
        self.height = {}
        self.weight = {}
        self.measurements = {}
        self.behavior = {
            "StartTime": {},
            "Duration": 0,
            "BehaviorId": 0,
            "MoveTarget": [],
        }
        self.gold = 0
        self.position = ["0"]
        self.classroom = []
        self.officeroom = ["0"]
        self.knowledge = {}
        self.language = {}
        self.mother_tongue = "Chinese"
        self.interest = {}
        self.dormitory = "0"
        self.birthday = {}
        self.weigt_tem = "Ordinary"
        self.bodyfat_tem = "Ordinary"
        self.bodyfat = {}
        self.sex_experience_tem = "None"
        self.clothing_tem = "Uniform"
        self.chest_tem = "Ordinary"
        self.chest = {}
        self.nature = {}
        self.status = {}
        self.put_on = {}
        self.wear_item = {}
        self.hit_point_tem = "Ordinary"
        self.mana_point_tem = "Ordinary"
        self.social_contact = {}
        self.occupation = ""

    def init_attr(self):
        """
        随机生成角色属性
        """
        self.language[self.mother_tongue] = 10000
        self.birthday = attr_calculation.get_rand_npc_birthday(self.age)
        self.end_age = attr_calculation.get_end_age(self.sex)
        self.height = attr_calculation.get_height(self.sex, self.age, {})
        bmi = attr_calculation.get_bmi(self.weigt_tem)
        self.weight = attr_calculation.get_weight(
            bmi, self.height["NowHeight"]
        )
        self.bodyfat = attr_calculation.get_bodyfat(self.sex, self.bodyfat_tem)
        self.measurements = attr_calculation.get_measurements(
            self.sex,
            self.height["NowHeight"],
            self.weight,
            self.bodyfat,
            self.bodyfat_tem,
        )
        self.sex_experience = attr_calculation.get_sex_experience(
            self.sex_experience_tem
        )
        self.sex_grade = attr_calculation.get_sex_grade(self.sex_experience)
        default_clothing_data = clothing.creator_suit(
            self.clothing_tem, self.sex
        )
        self.clothing = {
            clothing: {uuid.uuid1(): default_clothing_data[clothing]}
            if clothing in default_clothing_data
            else {}
            for clothing in self.clothing
        }
        self.chest = attr_calculation.get_chest(self.chest_tem, self.birthday)
        self.hit_point_max = attr_calculation.get_max_hit_point(
            self.hit_point_tem
        )
        self.hit_point = self.hit_point_max
        self.mana_point_max = attr_calculation.get_max_mana_point(
            self.mana_point_tem
        )
        self.mana_point = self.mana_point_max
        self.nature = nature.get_random_nature()
        self.status = text_loading.get_game_data(
            constant.FilePath.CHARACTER_STATE_PATH
        )
        self.wear_item = {
            "Wear": {
                key: {}
                for key in text_loading.get_game_data(
                    constant.FilePath.WEAR_ITEM_PATH
                )["Wear"]
            },
            "Item": {},
        }
        self.engraving = {
            "Pain": 0,
            "Happy": 0,
            "Yield": 0,
            "Fear": 0,
            "Resistance": 0,
        }
        self.social_contact = {
            social: {}
            for social in text_loading.get_text_data(
                constant.FilePath.STAGE_WORD_PATH, "144"
            )
        }
        self.init_class()
        self.put_on_clothing()
        if self.occupation == "":
            if self.age <= 18:
                self.occupation = "Student"
            else:
                self.occupation = "Teacher"

    def init_class(self):
        """
        初始化角色班级
        """
        if self.age <= 18 and self.age >= 7:
            class_grade = str(self.age - 6)
            self.classroom = random.choice(
                cache_contorl.place_data["Classroom_" + class_grade]
            )

    def put_on_clothing(self):
        """
        角色自动选择并穿戴服装
        Keyword arguments:
        character_id -- 角色服装数据
        """
        character_clothing_data = self.clothing
        collocation_data = {}
        clothings_name_data = clothing.get_clothing_name_data(
            character_clothing_data
        )
        clothings_price_data = clothing.get_clothing_price_data(
            character_clothing_data
        )
        for clothing_type in clothings_name_data:
            clothing_type_data = clothings_name_data[clothing_type]
            for clothing_name in clothing_type_data:
                clothing_name_data = clothing_type_data[clothing_name]
                clothing_id = list(clothing_name_data.keys())[-1]
                clothing_data = character_clothing_data[clothing_type][
                    clothing_id
                ]
                now_collocation_data = clothing.get_clothing_collocation_data(
                    clothing_data,
                    clothing_type,
                    clothings_name_data,
                    clothings_price_data,
                    character_clothing_data,
                )
                if now_collocation_data != "None":
                    now_collocation_data[clothing_type] = clothing_id
                    now_collocation_data["Price"] += clothings_price_data[
                        clothing_type
                    ][clothing_id]
                    collocation_data[clothing_id] = now_collocation_data
        collocation_price_data = {
            collocation: collocation_data[collocation]["Price"]
            for collocation in collocation_data
        }
        collocation_id = list(
            value_handle.sorted_dict_for_values(collocation_price_data).keys()
        )[-1]
        self.put_on = collocation_data[collocation_id]
