import random
import math
import uuid
from typing import List, Dict
from Script.Core import cache_contorl, value_handle, constant, game_type
from Script.Design import character
from Script.Config import game_config


def creator_suit(suit_id: int, sex: int) -> Dict[int, game_type.Clothing]:
    """
    创建套装
    Keyword arguments:
    suit_name -- 套装模板
    sex -- 性别模板
    Return arguments:
    Dict[int,game_type.Clothing] -- 套装数据 服装穿戴位置:服装数据
    """
    suit_data = game_config.config_clothing_suit_data[suit_id][sex]
    clothing_data = {}
    for clothing_id in suit_data:
        clothing = creator_clothing(clothing_id)
        clothing_data[clothing.wear] = clothing
    return clothing_data


def creator_clothing(clothing_tem_id: int) -> game_type.Clothing:
    """
    创建服装的基础函数
    Keyword arguments:
    clothing_tem_id -- 服装id
    Return arguments:
    game_type.Clothing -- 生成的服装数据
    """
    clothing_data = game_type.Clothing()
    clothing_data.uid = uuid.uuid4()
    clothing_data.sexy = random.randint(1, 1000)
    clothing_data.handsome = random.randint(1, 1000)
    clothing_data.elegant = random.randint(1, 1000)
    clothing_data.fresh = random.randint(1, 1000)
    clothing_data.sweet = random.randint(1, 1000)
    clothing_data.warm = random.randint(0, 30)
    clothing_data.price = sum(
        [
            clothing_data.__dict__[x]
            for x in clothing_data.__dict__
            if isinstance(clothing_data.__dict__[x], int)
        ]
    )
    clothing_data.cleanliness = 100
    clothing_data.evaluation = game_config.config_clothing_evaluate_list[
        math.floor(clothing_data.price / 480) - 1
    ]
    clothing_data.tem_id = clothing_tem_id
    clothing_data.wear = game_config.config_clothing_tem[clothing_tem_id].clothing_type
    return clothing_data


def get_clothing_name_data(
    clothings_data: Dict[int, Dict[uuid.UUID, game_type.Clothing]]
) -> Dict[int, Dict[int, Dict[uuid.UUID, int]]]:
    """
    按服装的配表id对服装进行分类，获取同类下各服装的价值数据
    Keyword arguments:
    clothings_data -- 要分类的所有服装数据 服装穿戴位置:服装id:服装数据
    Return arguments:
    Dict[int,Dict[int,Dict[uuid.UUID,int]]] -- 服装信息 服装穿戴位置:服装配表id:服装唯一id:服装价值
    """
    clothing_name_data = {}
    for clothing_type in clothings_data:
        clothing_type_data = clothings_data[clothing_type]
        clothing_name_data.setdefault(clothing_type, {})
        for clothing_id in clothing_type_data:
            clothing_data: game_type.Clothing = clothing_type_data[clothing_id]
            clothing_tem_id = clothing_data.uid
            clothing_name_data[clothing_type].setdefault(clothing_tem_id, {})
            clothing_name_data[clothing_type][clothing_tem_id][clothing_id] = (
                clothing_data.price + clothing_data.cleanliness
            )
        clothing_name_data[clothing_type] = {
            clothing_tem_id: value_handle.sorted_dict_for_values(
                clothing_name_data[clothing_type][clothing_tem_id]
            )
            for clothing_tem_id in clothing_name_data[clothing_type]
        }
    return clothing_name_data


def get_clothing_price_data(
    clothings_data: Dict[int, Dict[uuid.UUID, game_type.Clothing]]
) -> Dict[int, Dict[uuid.UUID, int]]:
    """
    获取每个类型的服装价值数据
    Keyword arguments:
    clothings_data -- 原始服装数据 服装穿戴位置:服装唯一id:服装数据
    Return arguments:
    Dict[int,Dict[uuid.UUID,int]] -- 服装价值数据 服装穿戴位置:服装唯一id:服装价值数据
    """
    return {
        clothing_type: {
            clothing: clothings_data[clothing_type][clothing].price
            + clothings_data[clothing_type][clothing].cleanliness
            for clothing in clothings_data[clothing_type]
        }
        for clothing_type in clothings_data
    }


def get_clothing_collocation_data(
    now_clothing_data: game_type.Clothing,
    clothing_name_data: Dict[int, Dict[int, Dict[uuid.UUID, int]]],
    clothing_price_data: Dict[int, Dict[uuid.UUID, int]],
    clothing_data: Dict[int, Dict[uuid.UUID, game_type.Clothing]],
) -> Dict[int, Dict[int, uuid.UUID]]:
    """
    获取服装的当前搭配数据
    Keyword arguments:
    now_clothing_data -- 当前服装原始数据
    clothing_name_data -- 按服装具体名字分类并按价值排序后的所有要搭配的服装数据 服装穿戴位置:服装id:服装配表id:服装唯一id:服装价值
    clothing_price_data -- 按服装类型分类并按价值排序后的所有要搭配的服装数据 服装穿戴位置:服装唯一id:服装价值数据
    clothing_data -- 所有要查询的服装数据 服装穿戴位置:服装唯一id:服装数据
    Return arguments:
    Dict[int,uuid.UUID] -- 服装搭配列表 穿搭位置:服装uuid or None:不能进行搭配
    """
    collocation_data = {"Price": 0}
    clothing_collocation_type_data = {}
    if now_clothing_data.tem_id in game_config.config_clothing_collocational_data:
        clothing_collocation_type_data = game_config.config_clothing_collocational_data[
            now_clothing_data.tem_id
        ]
    for collocation_type in clothing_data:
        if collocation_type not in clothing_collocation_type_data:
            if collocation_type == now_clothing_data.wear:
                continue
            usually_collocation = get_appoint_type_clothing_top(
                now_clothing_data,
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
        else:
            collocation_data[collocation_type] = ""
            now_collocation_list = clothing_collocation_type_data[clothing_type]
            now_collocation_data = now_collocation_list[0]
            if now_collocation_data.collocational == 1:
                precedence_list = [
                    now_add_clothing_data.tem_id for now_add_clothing_data in now_collocation_list
                ]
                now_clothing_name_data = clothing_name_data[clothing_type]
                precedence_collocation = get_appoint_names_clothing_top(
                    precedence_list, now_clothing_name_data
                )
                if precedence_collocation != "None":
                    collocation_data[now_collocation_data.clothing_type] = precedence_collocation
                    collocation_data["Price"] += clothing_price_data[now_collocation_data.clothing_type][
                        precedence_collocation
                    ]
                else:
                    usually_collocation = get_appoint_type_clothing_top(
                        now_clothing_data,
                        clothing_data,
                        clothing_price_data,
                        collocation_data,
                    )
                    if usually_collocation != "None":
                        collocation_data[collocation_type] = usually_collocation
                        collocation_data["Price"] += clothing_price_data[collocation_type][
                            usually_collocation
                        ]
                    else:
                        collocation_data = "None"
                        break
            elif now_collocation_data.collocational in {3, 4, 5}:
                precedence_list = [
                    now_add_clothing_data.tem_id for now_add_clothing_data in now_collocation_list
                ]
                now_clothing_name_data = clothing_name_data[clothing_type]
                precedence_collocation = get_appoint_names_clothing_top(
                    precedence_list, now_clothing_name_data
                )
                if precedence_collocation != "None":
                    collocation_data[now_collocation_data.clothing_type] = precedence_collocation
                    collocation_data["Price"] += clothing_price_data[now_collocation_data.clothing_type][
                        precedence_collocation
                    ]
                else:
                    collocation_data = "None"
                    break
    return collocation_data


def get_appoint_names_clothing_top(
    appoint_name_list: List[int],
    clothing_type_name_data: Dict[int, Dict[uuid.UUID, int]],
) -> uuid.UUID:
    """
    获取指定服装类型数据下指定的服装中价值最高的服装
    Keyword arguments:
    appoint_name_list -- 要获取的服装表id列表
    clothing_type_name_data -- 以表id为分类的已排序的要查询的服装数据 服装表id:服装uuid:服装价值
    Return arguments
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
    now_clothing_data: game_type.Clothing,
    clothing_data: Dict[int, Dict[uuid.UUID, game_type.Clothing]],
    clothing_price_data: Dict[int, Dict[uuid.UUID, int]],
    new_clothing_type: int,
    collocation_data: Dict[int, uuid.UUID],
) -> uuid.UUID:
    """
    获取指定类型下的可搭配的衣服中数值最高的衣服
    Keyword arguments:
    now_clothing_data -- 当前服装数据
    clothing_data -- 要查询的所有服装数据 服装穿戴位置:服装uuid:服装数据
    clothing_price_data -- 已按价值排序的各类型服装数据 服装穿戴位置:服装uuid:服装价值数据
    new_clothing_type -- 要查询的服装类型
    collocation_data -- 已有的穿戴数据 服装类型:服装uuid
    Return arguments:
    uuid.UUID -- 获取到的服装uuid
    """
    clothing_type_data = clothing_price_data[new_clothing_type]
    clothing_type_data_list = list(clothing_type_data.keys())
    if clothing_type_data_list != []:
        clothing_type_data_list.reverse()
    for new_clothing in clothing_type_data_list:
        new_clothing_data = clothing_data[new_clothing_type][new_clothing]
        return_judge = True
        if not judge_clothing_collocation(
            now_clothing_data,
            new_clothing_data,
        ):
            continue
        for collocation_type in collocation_data:
            if collocation_type == "Price":
                continue
            now_collocation_id = collocation_data[collocation_type]
            now_collocation_data = clothing_data[collocation_type][now_collocation_id]
            if not judge_clothing_collocation(now_collocation_data, new_clothing_data):
                return_judge = False
                break
        if not return_judge:
            continue
        return new_clothing
    return "None"


def judge_clothing_collocation(
    old_clothing_data: game_type.Clothing,
    new_clothing_data: game_type.Clothing,
) -> bool:
    """
    判断两件服装是否能够进行搭配
    Keyword arguments:
    old_clothing_data -- 旧服装数据
    new_clothing_data -- 新服装数据
    Return arguments:
    bool -- 搭配校验
    """
    old_clothing_data_restrict_data = {}
    if old_clothing_data.tem_id in game_config.config_clothing_collocational_data:
        old_clothing_data_restrict_data = game_config.config_clothing_collocational_data[
            old_clothing_data.tem_id
        ]
    new_clothing_data_restrict_data = {}
    if new_clothing_data.tem_id in game_config.config_clothing_collocational_data:
        new_clothing_data_restrict_data = game_config.config_clothing_collocational_data[
            new_clothing_data.tem_id
        ]
    if new_clothing_data.wear in old_clothing_data_restrict_data:
        old_judge = old_clothing_data_restrict_data[new_clothing_data.wear][0]
        if old_judge.collocational == 2:
            return 0
        elif old_judge.collocational in {3, 4, 5}:
            judge = 1
            for now_judge in old_clothing_data_restrict_data[new_clothing_data.wear]:
                if now_judge.clothing_tem == new_clothing_data.tem_id:
                    judge = 0
                    break
            if judge:
                return 0
    if old_clothing_data.wear in new_clothing_data_restrict_data:
        new_judge = new_clothing_data_restrict_data[old_clothing_data.wear][0]
        if new_judge == 2:
            return 0
        elif new_judge.collocational in {3, 4, 5}:
            judge = 1
            for now_judge in new_clothing_data_restrict_data[old_clothing_data.wear]:
                if now_judge.clothing_tem == old_clothing_data.tem_id:
                    judge = 0
                    break
            if judge:
                return 0
    return 1


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
