import os
import pickle
from typing import Dict, List
from Script.Core import (
    game_type, json_handle, get_text,
    text_handle, cache_control, dijkstra,
    game_path_config
)
from Script.Design import map_handle, constant

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
map_data_path = os.path.join("data", "map")
""" 地图配置数据路径 """
scene_path_edge_path = os.path.join(game_path_config.MAP_DATA_PATH, "ScenePath")
""" 预处理的所有寻路路径配置文件路径 """
all_scene_data_path = os.path.join(game_path_config.MAP_DATA_PATH, "SceneData")
""" 预处理的所有场景数据路径 """
all_place_data_path = os.path.join(game_path_config.MAP_DATA_PATH, "PlaceData")
""" 预处理的所有地点数据路径 """
all_map_data_path = os.path.join(game_path_config.MAP_DATA_PATH, "MapData")
""" 预处理的所有地图数据路径 """
all_move_time_path = os.path.join(game_path_config.MAP_DATA_PATH, "MoveTime")
""" 预处理的所有场景移动时间数据路径 """


def init_map_data():
    """载入地图和场景数据"""
    if (
        os.path.exists(all_scene_data_path)
        and os.path.exists(all_map_data_path)
        and os.path.exists(all_place_data_path)
        and os.path.exists(scene_path_edge_path)
        and os.path.exists(all_move_time_path)
    ):
        with open(all_scene_data_path, "rb") as all_scene_data_file:
            cache.scene_data = pickle.load(all_scene_data_file)
        with open(all_map_data_path, "rb") as all_map_data_file:
            cache.map_data = pickle.load(all_map_data_file)
        with open(all_place_data_path, "rb") as all_place_data_file:
            constant.place_data = pickle.load(all_place_data_file)
        map_handle.scene_move_time = json_handle.load_json(all_move_time_path)
        map_handle.scene_path_edge = json_handle.load_json(scene_path_edge_path)
    else:
        load_dir_now(map_data_path)
        with open(all_map_data_path, "wb") as all_map_data_file:
            pickle.dump(cache.map_data, all_map_data_file)
        with open(all_scene_data_path, "wb") as all_scene_data_file:
            pickle.dump(cache.scene_data, all_scene_data_file)
        with open(all_place_data_path, "wb") as all_place_data_file:
            pickle.dump(constant.place_data, all_place_data_file)
        map_handle.init_scene_edge_path_data()
        map_handle.init_move_time_data()
    for scene_path in cache.scene_data:
        scene_data = cache.scene_data[scene_path]
        if scene_data.in_door:
            constant.in_door_scene_list.append(scene_data.scene_path)


def load_dir_now(data_path: str):
    """
    获取路径下的地图数据
    Keyword arguments:
    data_path -- 地图路径
    """
    for i in os.listdir(data_path):
        now_path = os.path.join(data_path, i)
        if os.path.isfile(now_path):
            now_file = i.split(".")
            if len(now_file) > 1 and now_file[1] == "json":
                if now_file[0] == "Scene":
                    now_scene_data = game_type.Scene()
                    now_scene_data.scene_path = get_map_system_path_str(
                        get_map_system_path_for_path(now_path)
                    )
                    load_scene_data = json_handle.load_json(now_path)
                    now_scene_data.scene_name = get_text._(load_scene_data["SceneName"])
                    now_scene_data.in_door = load_scene_data["InOutDoor"] == "In"
                    now_scene_data.scene_tag = load_scene_data["SceneTag"]
                    cache.scene_data[now_scene_data.scene_path] = now_scene_data
                    constant.place_data.setdefault(now_scene_data.scene_tag, [])
                    constant.place_data[now_scene_data.scene_tag].append(now_scene_data.scene_path)
                elif now_file[0] == "Map":
                    now_map_data = game_type.Map()
                    now_map_data.map_path = get_map_system_path_str(
                        get_map_system_path_for_path(now_path)
                    )
                    with open(
                        os.path.join(data_path, "Map"), "r", encoding="utf-8"
                    ) as now_read_file:
                        draw_data = now_read_file.read()
                        now_map_data.map_draw = get_print_map_data(draw_data)
                    load_map_data = json_handle.load_json(now_path)
                    now_map_data.map_name = get_text._(load_map_data["MapName"])
                    now_map_data.path_edge = load_map_data["PathEdge"]
                    now_map_data.sorted_path = get_sorted_map_path_data(now_map_data.path_edge)
                    cache.map_data[now_map_data.map_path] = now_map_data
        else:
            load_dir_now(now_path)


def get_map_system_path_for_path(now_path: str) -> List[str]:
    """
    从地图文件路径获取游戏地图系统路径
    Keyword arguments:
    now_path -- 地图文件路径
    Return arguments:
    List[str] -- 游戏地图路径
    """
    current_dir = os.path.dirname(os.path.abspath(now_path))
    current_dir_str = str(current_dir)
    map_start_list = current_dir_str.split("map")
    current_dir_str = map_start_list[1]
    map_system_path = current_dir_str.split(os.sep)
    map_system_path = map_system_path[1:]
    return map_system_path


def get_map_system_path_str(now_path: List[str]) -> str:
    """
    将游戏地图系统路径转换为字符串
    Keyword arguments:
    now_path -- 游戏地图路径
    Return arguments:
    str -- 地图路径字符串
    """
    return os.sep.join(now_path)


def get_print_map_data(map_draw: str) -> game_type.MapDraw:
    """
    获取绘制地图的富文本和按钮数据
    Keyword arguments:
    map_draw -- 绘制地图的原始数据
    Return arguments:
    game_type.MapDraw -- 地图绘制数据
    """
    map_y_list = map_draw.split("\n")
    map_draw_data = game_type.MapDraw()
    for map_x_list_id, _unused in enumerate(map_y_list):
        set_map_button = False
        map_x_list = map_y_list[map_x_list_id]
        now_draw_list = game_type.MapDrawLine()
        new_x_list = ""
        now_cmd = ""
        i = 0
        while i in range(len(map_x_list)):
            if not set_map_button and map_x_list[i : i + 11] != "<mapbutton>":
                new_x_list += map_x_list[i]
            elif not set_map_button and map_x_list[i : i + 11] == "<mapbutton>":
                i += 10
                set_map_button = 1
                if new_x_list:
                    now_draw = game_type.MapDrawText()
                    now_draw.text = new_x_list
                    now_draw_list.draw_list.append(now_draw)
                    now_draw_list.width += text_handle.get_text_index(new_x_list)
                    new_x_list = ""
            elif set_map_button and map_x_list[i : i + 12] != "</mapbutton>":
                now_cmd += map_x_list[i]
            else:
                set_map_button = 0
                now_draw = game_type.MapDrawText()
                now_draw.text = now_cmd
                now_draw.is_button = 1
                now_draw_list.draw_list.append(now_draw)
                now_draw_list.width += text_handle.get_text_index(now_cmd)
                now_cmd = ""
                i += 11
            i += 1
        if new_x_list:
            now_draw = game_type.MapDrawText()
            now_draw.text = new_x_list
            now_draw_list.draw_list.append(now_draw)
            now_draw_list.width += text_handle.get_text_index(new_x_list)
        map_draw_data.draw_text.append(now_draw_list)
    return map_draw_data


def get_sorted_map_path_data(
    map_data: Dict[str, Dict[str, int]]
) -> Dict[str, Dict[str, game_type.TargetPath]]:
    """
    获取地图下各节点到目标节点的最短路径数据
    Keyword arguments:
    map_data -- 地图节点数据 当前节点:可通行节点:所需时间
    Return arguments:
    Dict[int,Dict[int,game_type.TargetPath]] -- 最短路径数据 当前节点:目标节点:路径对象
    """
    sorted_path_data = {}
    for node in map_data:
        node_path = dijkstra.dijkstra(map_data, node)
        sorted_path_data.setdefault(node,{})
        for target_id in node_path:
            target = node_path[target_id]
            target_path = game_type.TargetPath()
            target_path.path = target.path[1:]
            target_path.time = target.path_times[1:]
            sorted_path_data[node][target.node_id] = target_path
    return sorted_path_data
