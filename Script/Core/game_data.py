# -*- coding: UTF-8 -*-
import os
import pickle
import platform
from dijkstar import Graph, find_path
from Script.Core.game_path_config import game_path
from Script.Core import json_handle, cache_contorl, value_handle

game_data = {}
scene_data = {}
map_data = {}


def load_dir_now(data_path: str):
    """
    获取路径下的游戏数据
    Keyword arguments:
    data_path -- 要载入数据的路径
    """
    now_data = {}
    if os.listdir(data_path):
        for i in os.listdir(data_path):
            now_path = os.path.join(data_path, i)
            if os.path.isfile(now_path):
                now_file = i.split(".")
                if len(now_file) > 1:
                    if now_file[1] == "json":
                        if now_file[0] == "Scene":
                            now_scene_data = {}
                            map_system_path = get_map_system_path_for_path(
                                now_path
                            )
                            map_system_path_str = get_map_system_path_str(
                                map_system_path
                            )
                            load_scene_data = json_handle.load_json(now_path)
                            now_scene_data.update(load_scene_data)
                            now_scene_data["SceneCharacterData"] = {}
                            now_scene_data["ScenePath"] = map_system_path
                            now_scene_data = {
                                map_system_path_str: now_scene_data
                            }
                            scene_data.update(now_scene_data)
                            now_scene_tag = load_scene_data["SceneTag"]
                            if now_scene_tag not in cache_contorl.place_data:
                                cache_contorl.place_data[now_scene_tag] = []
                            cache_contorl.place_data[now_scene_tag].append(
                                map_system_path_str
                            )
                        elif now_file[0] == "Map":
                            now_map_data = {}
                            map_system_path = get_map_system_path_for_path(
                                now_path
                            )
                            now_map_data["MapPath"] = map_system_path
                            with open(
                                os.path.join(data_path, "Map"), "r"
                            ) as now_read_file:
                                draw_data = now_read_file.read()
                                now_map_data["MapDraw"] = get_print_map_data(
                                    draw_data
                                )
                            map_system_path_str = get_map_system_path_str(
                                map_system_path
                            )
                            now_map_data.update(
                                json_handle.load_json(now_path)
                            )
                            cache_contorl.now_init_map_id = map_system_path_str
                            sorted_path_data = get_sorted_map_path_data(
                                now_map_data["PathEdge"]
                            )
                            now_map_data["SortedPath"] = sorted_path_data
                            map_data[map_system_path_str] = now_map_data
                        else:
                            if now_file[0] == "NameIndex":
                                data = json_handle.load_json(now_path)
                                init_name_region(data["Boys"], 0)
                                init_name_region(data["Girls"], 1)
                            elif now_file[0] == "FamilyIndex":
                                data = json_handle.load_json(now_path)
                                init_name_region(data["FamilyNameList"], 2)
                            else:
                                now_data[now_file[0]] = json_handle.load_json(
                                    now_path
                                )
                                if now_file[0] == "Equipment":
                                    init_clothing_data(
                                        now_data[now_file[0]]["Clothing"]
                                    )
                                elif now_file[0] == "StatureDescription":
                                    init_stature_description(
                                        now_data[now_file[0]]["Priority"]
                                    )
                                elif now_file[0] == "WearItem":
                                    init_wear_item_type_data(
                                        now_data[now_file[0]]["Item"]
                                    )
            else:
                now_data[i] = load_dir_now(now_path)
    return now_data


def init_stature_description(stature_descrition_data: dict):
    """
    初始化身材描述文本权重数据
    Keyword arguments:
    sd_data -- 身材描述文本数据
    """
    cache_contorl.stature_descrition_priorition_data = {
        priority: {
            i: len(stature_descrition_data[priority][i]["Condition"])
            for i in range(len(stature_descrition_data[priority]))
        }
        for priority in range(len(stature_descrition_data))
    }


def init_wear_item_type_data(wear_item_data: dict):
    """
    初始化可穿戴道具类型数据
    Keyword argumenys:
    wear_item_data -- 可穿戴道具数据
    """
    cache_contorl.wear_item_type_data = {
        wear: {
            item: 1
            for item in wear_item_data
            if wear in wear_item_data[item]["Wear"]
        }
        for item in wear_item_data
        for wear in wear_item_data[item]["Wear"]
    }


def init_name_region(name_data: dict, man_judge: int):
    """
    初始化性别名字随机权重
    Keyword arguments:
    name_data -- 名字数据
    man_judge -- 类型校验(0:男,1:女,2:姓)
    """
    region_list = value_handle.get_region_list(name_data)
    if man_judge == 0:
        cache_contorl.boys_region_list = region_list
        cache_contorl.boys_region_int_list = list(map(int, region_list))
    elif man_judge == 1:
        cache_contorl.girls_region_list = region_list
        cache_contorl.girls_region_int_list = list(map(int, region_list))
    else:
        cache_contorl.family_region_list = region_list
        cache_contorl.family_region_int_list = list(map(int, region_list))


def get_sorted_map_path_data(map_data: dict) -> dict:
    """
    获取地图下各节点到目标节点的最短路径数据
    Keyword arguments:
    map_data -- 地图节点数据
    """
    graph = Graph()
    sorted_path_data = {}
    for node in map_data.keys():
        for target in map_data[node]:
            graph.add_edge(node, target, {"cost": map_data[node][target]})
    cost_func = lambda u, v, e, prev_e: e["cost"]
    for node in map_data.keys():
        new_data = {node: {}}
        for target in map_data.keys():
            if target != node:
                find_path_data = find_path(
                    graph, node, target, cost_func=cost_func
                )
                new_data[node].update(
                    {
                        target: {
                            "Path": find_path_data.nodes[1:],
                            "Time": find_path_data.costs,
                        }
                    }
                )
        sorted_path_data.update(new_data)
    return sorted_path_data


def get_map_system_path_for_path(now_path: str) -> list:
    """
    从地图文件路径获取游戏地图系统路径
    Keyword arguments:
    now_path -- 地图文件路径
    """
    current_dir = os.path.dirname(os.path.abspath(now_path))
    current_dir_str = str(current_dir)
    map_start_list = current_dir_str.split("map")
    current_dir_str = map_start_list[1]
    map_system_path = current_dir_str.split(os.sep)
    map_system_path = map_system_path[1:]
    return map_system_path


def get_map_system_path_str(now_path: list) -> str:
    """
    将游戏地图系统路径转换为字符串
    """
    return os.sep.join(now_path)


def get_print_map_data(map_draw: str) -> dict:
    """
    获取绘制地图的富文本和按钮数据
    Keyword arguments:
    map_draw -- 绘制地图的原始数据
    """
    map_y_list = map_draw.split("\n")
    new_map_y_list = []
    map_x_list_cmd_data = {}
    map_x_cmd_id_data = {}
    for map_x_list_id in range(len(map_y_list)):
        set_map_button = False
        map_x_list = map_y_list[map_x_list_id]
        map_x_list_cmd_list = []
        cmd_id_list = []
        new_x_list = ""
        now_cmd = ""
        i = 0
        while i in range(len(map_x_list)):
            if not set_map_button and map_x_list[i : i + 11] != "<mapbutton>":
                new_x_list += map_x_list[i]
            elif (
                not set_map_button and map_x_list[i : i + 11] == "<mapbutton>"
            ):
                i += 10
                set_map_button = True
            elif set_map_button and map_x_list[i : i + 12] != "</mapbutton>":
                now_cmd += map_x_list[i]
            else:
                set_map_button = False
                map_x_list_cmd_list.append(now_cmd)
                cmd_id_list.append(len(new_x_list))
                now_cmd = ""
                i += 11
            i += 1
        map_x_list_cmd_data[map_x_list_id] = map_x_list_cmd_list
        new_map_y_list.append(new_x_list)
        map_x_cmd_id_data[map_x_list_id] = cmd_id_list
    return {
        "Draw": new_map_y_list,
        "Cmd": map_x_list_cmd_data,
        "CmdId": map_x_cmd_id_data,
    }


def get_path_list(root_path: str) -> list:
    """
    获取路径下所有子目录列表
    Keyword arguments:
    root_path -- 要获取的目录所在根路径
    """
    return [
        name
        for name in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, name))
    ]


def init_clothing_data(original_clothing_data: dict):
    """
    初始化服装类型数据
    """
    clothing_type_data = {
        x: get_original_clothing(j, k, original_clothing_data[j][k][x])
        for j in original_clothing_data
        for k in original_clothing_data[j]
        for x in original_clothing_data[j][k]
    }
    cache_contorl.clothing_type_data = clothing_type_data


def get_original_clothing(
    clothing_type: str, clothing_sex: str, now_clothing_data: dict
):
    """
    生成无分类的原始服装数据
    Keyword arguments:
    clothing_type -- 服装类型
    clothing_sex -- 服装性别倾向
    clothing_data -- 原始服装数据
    """
    now_clothing_data["Type"] = clothing_type
    now_clothing_data["Sex"] = clothing_sex
    return now_clothing_data


def init_data_json():
    """
    初始化游戏数据文件
    """
    data_dir = os.path.join(game_path, "data")
    data_path = os.path.join(game_path, "data.json")
    f = open(data_path, "wb")
    game_data.update(load_dir_now(data_dir))
    now_data = {
        "gamedata": game_data,
        "placedata": cache_contorl.place_data,
        "staturedata": cache_contorl.stature_descrition_priorition_data,
        "clothingdata": cache_contorl.clothing_type_data,
        "scenedata": scene_data,
        "mapdata": map_data,
        "boysregiondata": cache_contorl.boys_region_int_list,
        "girlsregiondata": cache_contorl.girls_region_int_list,
        "familyregiondata": cache_contorl.family_region_int_list,
        "boysdata": cache_contorl.boys_region_list,
        "girlsdata": cache_contorl.girls_region_list,
        "familydata": cache_contorl.family_region_list,
        "weardata": cache_contorl.wear_item_type_data,
        "system": platform.system(),
    }
    pickle.dump(now_data, f)


def init():
    """
    初始化游戏数据
    """
    data_path = os.path.join(game_path, "data.json")
    if os.path.exists(data_path):
        f = open(data_path, "rb")
        data = pickle.load(f)
        if "system" in data and data["system"] == platform.system():
            game_data.update(data["gamedata"])
            cache_contorl.place_data = data["placedata"]
            cache_contorl.stature_descrition_priorition_data = data[
                "staturedata"
            ]
            cache_contorl.clothing_type_data = data["clothingdata"]
            scene_data.update(data["scenedata"])
            map_data.update(data["mapdata"])
            cache_contorl.boys_region_int_list = data["boysregiondata"]
            cache_contorl.girls_region_int_list = data["girlsregiondata"]
            cache_contorl.family_region_int_list = data["familyregiondata"]
            cache_contorl.boys_region_list = data["boysdata"]
            cache_contorl.girls_region_list = data["girlsdata"]
            cache_contorl.family_region_list = data["familydata"]
            cache_contorl.wear_item_type_data = data["weardata"]
        else:
            init_data_json()
    else:
        init_data_json()
