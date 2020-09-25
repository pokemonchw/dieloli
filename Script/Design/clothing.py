import random
import math
from Script.Core import cache_contorl, text_loading, value_handle, constant
from Script.Design import character

clothing_tag_text_list = {
    "Sexy": "性感",
    "Handsome": "帅气",
    "Elegant": "典雅",
    "Fresh": "清新",
    "Sweet": "可爱",
    "Warm": "保暖",
    "Cleanliness": "清洁",
}
clothing_type_text_list = {
    "Coat": "外套",
    "Underwear": "上衣",
    "Pants": "裤子",
    "Skirt": "裙子",
    "Shoes": "鞋子",
    "Socks": "袜子",
    "Bra": "胸罩",
    "Underpants": "内裤",
}


def creator_suit(suit_name: str, sex: str) -> dict:
    """
    创建套装
    Keyword arguments:
    suit_name -- 套装模板
    sex -- 性别模板
    """
    suit_data = text_loading.get_text_data(constant.FilePath.EQUIPMENT_PATH, "Suit")[suit_name][sex]
    new_suit_data = {
        clothing: creator_clothing(suit_data[clothing])
        for clothing in suit_data
        if suit_data[clothing] != ""
    }
    return new_suit_data


def creator_clothing(clothing_name: str) -> dict:
    """
    创建服装的基础函数
    Keyword arguments:
    clothing_name -- 服装名字
    """
    clothing_data = {}
    clothing_data["Sexy"] = random.randint(1, 1000)
    clothing_data["Handsome"] = random.randint(1, 1000)
    clothing_data["Elegant"] = random.randint(1, 1000)
    clothing_data["Fresh"] = random.randint(1, 1000)
    clothing_data["Sweet"] = random.randint(1, 1000)
    clothing_data["Warm"] = random.randint(0, 30)
    clothing_data["Price"] = sum([clothing_data[x] for x in clothing_data])
    set_clothint_evaluation_text(clothing_data)
    clothing_data["Cleanliness"] = 100
    clothing_data.update(cache_contorl.clothing_type_data[clothing_name])
    return clothing_data


def get_clothing_name_data(clothings_data: dict) -> dict:
    """
    按服装的具体名称对服装进行分类，获取同类下各服装的价值数据
    Keyword arguments:
    clothings_data -- 要分类的所有服装数据
    """
    clothing_name_data = {}
    for clothing_type in clothings_data:
        clothing_type_data = clothings_data[clothing_type]
        clothing_name_data.setdefault(clothing_type, {})
        for clothing in clothing_type_data:
            clothing_data = clothing_type_data[clothing]
            clothing_name = clothing_data["Name"]
            clothing_name_data[clothing_type].setdefault(clothing_name, {})
            clothing_name_data[clothing_type][clothing_name][clothing] = (
                clothing_data["Price"] + clothing_data["Cleanliness"]
            )
        clothing_name_data[clothing_type] = {
            clothing_name: value_handle.sorted_dict_for_values(
                clothing_name_data[clothing_type][clothing_name]
            )
            for clothing_name in clothing_name_data[clothing_type]
        }
    return clothing_name_data


def get_clothing_price_data(clothings_data: dict) -> dict:
    """
    为每个类型的服装进行排序
    Keyword arguments:
    clothings_data -- 要排序的所有服装数据
    """
    return {
        clothing_type: {
            clothing: clothings_data[clothing_type][clothing]["Price"]
            + clothings_data[clothing_type][clothing]["Cleanliness"]
            for clothing in clothings_data[clothing_type]
        }
        for clothing_type in clothings_data
    }


def get_clothing_collocation_data(
    now_clothing_data: dict,
    now_clothing_type: str,
    clothing_name_data: dict,
    clothing_price_data: dict,
    clothing_data: dict,
):
    """
    获取服装的当前搭配数据
    Keyword arguments:
    now_clothing_data -- 当前服装原始数据
    now_clothing_type -- 服装类型
    clothing_name_data -- 按服装具体名字分类并按价值排序后的所有要搭配的服装数据
    clothing_price_data -- 按服装类型分类并按价值排序后的所有要搭配的服装数据
    clothing_data -- 所有要查询的服装数据
    """
    collocation_data = {"Price": 0}
    clothing_collocation_type_data = now_clothing_data["CollocationalRestriction"]
    for collocation_type in clothing_data:
        collocation_data[collocation_type] = ""
        if collocation_type not in clothing_collocation_type_data:
            continue
        now_restrict = clothing_collocation_type_data[collocation_type]
        if now_restrict == "Precedence":
            clothing_now_type_precedence_list = now_clothing_data["Collocation"][collocation_type]
            precedence_collocation = get_appoint_names_clothing_top(
                list(clothing_now_type_precedence_list.keys()), clothing_name_data[collocation_type],
            )
            if precedence_collocation != "None":
                collocation_data[collocation_type] = precedence_collocation
                collocation_data["Price"] += clothing_price_data[collocation_type][precedence_collocation]
            else:
                usually_collocation = get_appoint_type_clothing_top(
                    now_clothing_data,
                    now_clothing_type,
                    clothing_data,
                    clothing_price_data,
                    collocation_type,
                    collocation_data,
                )
                if usually_collocation != "None":
                    collocation_data[collocation_type] = usually_collocation
                    collocation_data["Price"] += clothing_price_data[collocation_type][usually_collocation]
                else:
                    collocation_data = "None"
                    break
        elif now_restrict == "Usually":
            usually_collocation = get_appoint_type_clothing_top(
                now_clothing_data,
                now_clothing_type,
                clothing_data,
                clothing_price_data,
                collocation_type,
                collocation_data,
            )
            if usually_collocation != "None":
                collocation_data[collocation_type] = usually_collocation
                collocation_data["Price"] += clothing_price_data[collocation_type][usually_collocation]
            else:
                collocation_data[collocation_type] = ""
        elif now_restrict == "Must" or "Ornone":
            clothing_now_type_precedence_list = now_clothing_data["Collocation"][collocation_type]
            precedence_collocation = get_appoint_names_clothing_top(
                list(clothing_now_type_precedence_list.keys()), clothing_name_data[collocation_type],
            )
            if precedence_collocation != "None":
                collocation_data[collocation_type] = precedence_collocation
                collocation_data["Price"] += clothing_price_data[collocation_type][precedence_collocation]
            else:
                collocation_data = "None"
                break
    return collocation_data


def get_appoint_names_clothing_top(appoint_name_list: list, clothing_type_name_data: dict) -> str:
    """
    获取指定服装类型数据下指定名称的服装中价值最高的服装
    Keyword arguments:
    appoint_name_list -- 要获取的服装名字列表
    clothing_typeNameData -- 以名字为分类的已排序的要查询的服装数据
    """
    clothing_data = {
        list(clothing_type_name_data[appoint].keys())[-1]: clothing_type_name_data[appoint][
            list(clothing_type_name_data[appoint].keys())[-1]
        ]
        for appoint in appoint_name_list
        if appoint in clothing_type_name_data
    }
    if clothing_data != {}:
        return list(value_handle.sorted_dict_for_values(clothing_data).keys())[-1]
    return "None"


def get_appoint_type_clothing_top(
    now_clothing_data: str,
    now_clothing_type: str,
    clothing_data: dict,
    clothing_price_data,
    new_clothing_type: str,
    collocation_data: dict,
) -> str:
    """
    获取指定类型下的可搭配的衣服中数值最高的衣服
    Keyword arguments:
    now_clothing_name -- 当前服装名字
    now_clothing_type -- 当前服装类型
    clothing_data -- 要查询的所有服装数据
    clothing_price_data -- 已按价值排序的各类型服装数据
    new_clothing_type -- 要查询的服装类型
    collocation_data -- 已有的穿戴数据
    """
    clothing_type_data = clothing_price_data[new_clothing_type]
    clothing_type_data_list = list(clothing_type_data.keys())
    if clothing_type_data_list != []:
        clothing_type_data_list.reverse()
    for new_clothing in clothing_type_data_list:
        new_clothing_data = clothing_data[new_clothing_type][new_clothing]
        return_judge = True
        if not judge_clothing_collocation(
            now_clothing_data, now_clothing_type, new_clothing_data, new_clothing_type,
        ):
            continue
        for collocation_type in collocation_data:
            if collocation_type == "Price":
                continue
            now_collocation_id = collocation_data[collocation_type]
            if now_collocation_id == "":
                continue
            now_collocation_data = clothing_data[collocation_type][now_collocation_id]
            if not judge_clothing_collocation(
                now_collocation_data, collocation_type, new_clothing_data, new_clothing_type,
            ):
                return_judge = False
                break
        if not return_judge:
            continue
        return new_clothing
    return "None"


def judge_clothing_collocation(
    old_clothing_data: dict, old_clothing_type: str, new_clothing_data: dict, new_clothing_type: str,
) -> bool:
    """
    判断两件服装是否能够进行搭配
    Keyword arguments:
    old_clothing_data -- 旧服装数据
    old_clothing_type -- 旧服装类型
    new_clothing_data -- 新服装数据
    new_clothing_type -- 新服装类型
    """
    old_clothing_data_restrict_data = old_clothing_data["CollocationalRestriction"]
    new_clothing_data_restrict_data = new_clothing_data["CollocationalRestriction"]
    old_judge = old_clothing_data_restrict_data[new_clothing_type]
    new_judge = new_clothing_data_restrict_data[old_clothing_type]
    if old_judge in {"Must": 0, "Ornone": 1}:
        old_collocation_type_data = old_clothing_data["Collocation"][new_clothing_type]
        if new_clothing_data["Name"] not in old_collocation_type_data:
            return False
    elif old_judge == "None":
        return False
    if new_judge in {"Must": 0, "Ornone": 1}:
        new_collocation_type_data = new_clothing_data["Collocation"][old_clothing_type]
        if old_clothing_data["Name"] not in new_collocation_type_data:
            return False
    elif new_judge == "None":
        return False
    return True


clothing_evaluation_text_list = [
    text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, str(k)) for k in range(102, 112)
]
clothing_tagList = [
    text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, str(k)) for k in range(112, 117)
]


def set_clothint_evaluation_text(clothing_data: dict):
    """
    设置服装的评价文本
    Keyword arguments:
    clothing_data -- 服装数据
    """
    clothing_attr_data = [
        clothing_data["Sexy"],
        clothing_data["Handsome"],
        clothing_data["Elegant"],
        clothing_data["Fresh"],
        clothing_data["Sweet"],
    ]
    clothing_evaluation_text = clothing_evaluation_text_list[math.floor(clothing_data["Price"] / 480) - 1]
    clothing_tag_text = clothing_tagList[clothing_attr_data.index(max(clothing_attr_data))]
    clothing_data["Evaluation"] = clothing_evaluation_text
    clothing_data["Tag"] = clothing_tag_text


def init_character_clothing_put_on(player_pass=True):
    """
    为所有角色穿衣服
    Keyword arguments:
    player_pass -- 跳过主角 (default:True)
    """
    for character_id in cache_contorl.character_data:
        if player_pass and character_id == 0:
            continue
        character.put_on_clothing(cache_contorl.character_data[character_id])
