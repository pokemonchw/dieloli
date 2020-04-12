import pysnooper
from Script.Core import cache_contorl, text_loading, era_print, py_cmd, text_handle
from Script.Design import map_handle, cmd_button_queue

panel_state_text_data = text_loading.get_text_data(text_loading.CMD_PATH, "cmdSwitch")
panel_state_on_text = panel_state_text_data[1]
panel_state_off_text = panel_state_text_data[0]


def see_map_panel() -> list:
    """
    地图绘制面板
    """
    input_s = []
    title_text = text_loading.get_text_data(text_loading.STAGE_WORD_PATH, "78")
    now_map = cache_contorl.now_map
    now_map_map_system_str = map_handle.get_map_system_path_str_for_list(now_map)
    map_name = cache_contorl.map_data[now_map_map_system_str]["MapName"]
    era_print.little_title_print(title_text + ": " + map_name + " ")
    input_s = input_s + map_handle.print_map(now_map)
    return input_s


def see_move_path_panel() -> dict:
    """
    当前场景可直接通往的移动路径绘制面板
    """
    input_s = []
    now_scene = cache_contorl.character_data["character"][0].position
    now_map = cache_contorl.now_map
    now_map_str = map_handle.get_map_system_path_str_for_list(now_map)
    map_data = cache_contorl.map_data[now_map_str]
    move_path_info = text_loading.get_text_data(text_loading.MESSAGE_PATH, "27")
    era_print.normal_print("\n")
    era_print.line_feed_print(move_path_info)
    path_edge = map_data["PathEdge"]
    map_scene_id = str(map_handle.get_map_scene_id_for_scene_path(now_map, now_scene))
    scene_path = path_edge[map_scene_id]
    scene_path_list = list(scene_path.keys())
    if map_scene_id in scene_path_list:
        remove(map_scene_id)
    if len(scene_path_list) > 0:
        scene_cmd = []
        for scene in scene_path_list:
            now_map_str = map_handle.get_map_system_path_str_for_list(now_map)
            load_scene_data = map_handle.get_scene_data_for_map(now_map_str, scene)
            scene_name = load_scene_data["SceneName"]
            scene_cmd.append(scene_name)
        yrn = cmd_button_queue.option_str(
            cmd_list=None,
            cmd_list_data=scene_cmd,
            cmd_column=4,
            askfor=False,
            cmd_size="center",
        )
        input_s = input_s + yrn
    else:
        error_move_text = text_loading.get_text_data(text_loading.MESSAGE_PATH, "28")
        era_print.normal_print(error_move_text)
    era_print.restart_line_print()
    return {"input_s": input_s, "scene_path_list": scene_path_list}


def show_scene_name_list_panel() -> str:
    """
    地图下场景名称绘制面板
    """
    title_text = text_loading.get_text_data(text_loading.STAGE_WORD_PATH, "86")
    era_print.normal_print(title_text)
    panel_state = cache_contorl.panel_state["SeeSceneNameListPanel"]
    if panel_state == "0":
        py_cmd.pcmd(panel_state_off_text, "SeeSceneNameListPanel")
        era_print.line_feed_print()
        now_map = cache_contorl.now_map
        now_position = cache_contorl.character_data["character"][0].position
        now_scene = map_handle.get_scene_id_in_map_for_scene_path_on_map_path(
            now_position, now_map
        )
        now_map_map_system_str = map_handle.get_map_system_path_str_for_list(now_map)
        scene_name_data = map_handle.get_scene_name_list_for_map_path(
            now_map_map_system_str
        )
        scene_name_list = {}
        null_scene = now_scene
        for scene in scene_name_data:
            scene_name_list[scene] = scene + ":" + scene_name_data[scene]
        cmd_button_queue.option_str(
            None,
            4,
            "center",
            False,
            False,
            list(scene_name_list.values()),
            null_scene,
            list(scene_name_list.keys()),
        )
    else:
        py_cmd.pcmd(panel_state_on_text, "SeeSceneNameListPanel")
        era_print.line_feed_print()
    era_print.little_line_print()
    return "SeeSceneNameListPanel"


def back_scene_panel(start_id: str) -> list:
    """
    查看场景页面基础命令绘制面板
    Keyword arguments:
    start_id -- 面板命令起始id
    """
    see_map_cmd = []
    now_position = cache_contorl.character_data["character"][0].position
    now_map = map_handle.get_map_for_path(now_position)
    cmd_data = text_loading.get_text_data(
        text_loading.CMD_PATH, cmd_button_queue.SEE_MAP
    )
    see_map_cmd.append(cmd_data[0])
    if now_map != [] and cache_contorl.now_map != []:
        see_map_cmd.append(cmd_data[1])
    if now_map != cache_contorl.now_map:
        see_map_cmd.append(cmd_data[2])
    map_cmd_list = cmd_button_queue.option_int(
        cmd_list=None,
        cmd_list_data=see_map_cmd,
        cmd_column=3,
        askfor=False,
        cmd_size="center",
        start_id=start_id,
    )
    return map_cmd_list
