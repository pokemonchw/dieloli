import os
from typing import Dict
from dijkstar import Graph, find_path
from Script.Core import game_type,json_handle,cache_contorl,get_text


map_data_path = os.path.join("data","map")
""" 地图配置数据路径 """

def init_map_data():
    """ 载入地图和场景数据 """
    load_dir_now(map_data_path)


def load_dir_now(data_path:str):
    """
    获取路径下的地图数据
    Keyword arguments:
    data_path -- 地图路径
    """
    for i in os.listdir(data_path):
        now_path = os.path.join(data_path,i)
        if os.path.isfile(now_path):
            now_file = i.split(".")
            if len(now_file) > 1:
                if now_file[1] == "json":
                    if now_file[0] == "Scene":
                        now_scene_data = game_type.Scene()
                        now_scene_data.scene_path = get_map_system_path_str(get_map_system_path_for_path(now_path))
                        load_scene_data = json_handle.load_json(now_file)
                        now_scene_data.scene_name = get_text._(load_scene_data["SceneName"])
                        now_scene_data.in_door = (load_scene_data["InOutDoor"] == "In")
                        now_scene_data.scene_tag = load_scene_data["SceneTag"]
                        cache_contorl.scene_data[now_scene_data.scene_path] = now_scene_data
                        cache_contorl.place_data.setdefault(now_scene_data.scene_tag,[])
                        cache_contorl.place_data[now_scene_data.scene_tag].append(now_scene_data.scene_path)
                    elif now_file[0] == "Map":
                        now_map_data = game_type.Map()
                        now_map_data.map_path = get_map_system_path_str(get_map_system_path_for_path(now_path))
                        with open(os.path.join(data_path,"Map","r")) as now_read_file:
                            draw_data = now_read_file.read()
                            now_map_data.map_draw = get_print_map_data(draw_data)
                        load_map_data = json_handle.load_json(now_file)
                        now_map_data.map_name = load_map_data["MapName"]
                        now_map_data.path_edge = load_map_data["PathEdge"]
                        now_map_data.sorted_path = get_sorted_map_path_data(now_map_data.path_edge)
                        cache_contorl.map_data[now_map_data.map_path] = now_map_data

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

def get_map_system_path_str(now_path: List[str]) -> str:
    """
    将游戏地图系统路径转换为字符串
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
            elif not set_map_button and map_x_list[i : i + 11] == "<mapbutton>":
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
        map_draw_data.draw_text.append(new_x_list)
        map_draw_data.cmd[map_x_list_id] = map_x_list_cmd_list
        map_draw_data.cmd_id[map_x_list_id] = cmd_id_list
    return map_draw_data

def get_sorted_map_path_data(map_data: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, game_type.TargetPath]]:
    """
    获取地图下各节点到目标节点的最短路径数据
    Keyword arguments:
    map_data -- 地图节点数据 当前节点:可通行节点:所需时间
    Return arguments:
    Dict[int,Dict[int,game_type.TargetPath]] -- 最短路径数据 当前节点:目标节点:路径对象
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
                find_path_data = find_path(graph, node, target, cost_func=cost_func)
                target_path = game_type.TargetPath()
                target_path.path = find_path_data.nodes[1:]
                target_path.time = find_path_data.costs
                new_data[node][target] =  target_path
        sorted_path_data.update(new_data)
    return sorted_path_data
