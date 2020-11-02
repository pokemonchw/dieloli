from Script.Core import flow_handle, cache_contorl, py_cmd
from Script.Design import character_move, map_handle, panel_state_handle
from Script.Panel import see_map_panel


def see_map_flow():
    """
    地图查看流程
    """
    while True:
        py_cmd.clr_cmd()
        input_s = []
        map_cmd = see_map_panel.see_map_panel()
        start_id_1 = len(map_cmd)
        input_s = input_s + map_cmd
        move_path_cmd_data = see_map_panel.see_move_path_panel()
        move_path_cmd = move_path_cmd_data["input_s"]
        move_path_list = move_path_cmd_data["scene_path_list"]
        show_scene_name_list_cmd = see_map_panel.show_scene_name_list_panel()
        see_map_cmd = see_map_panel.back_scene_panel(start_id_1)
        input_s += see_map_cmd + move_path_cmd + [show_scene_name_list_cmd]
        yrn = flow_handle.askfor_all(input_s)
        back_button = str(start_id_1)
        now_position = cache_contorl.character_data[0].position
        now_position_map = map_handle.get_map_for_path(now_position)
        up_map_button = "Null"
        down_map_button = "Null"
        if now_position_map != [] and cache_contorl.now_map != []:
            up_map_button = str(int(start_id_1) + 1)
        if now_position_map != cache_contorl.now_map:
            if up_map_button == "Null":
                down_map_button = str(int(start_id_1) + 1)
            else:
                down_map_button = str(int(start_id_1) + 2)
        now_map = cache_contorl.now_map.copy()
        if yrn in map_cmd:
            now_target_path = map_handle.get_scene_path_for_map_scene_id(now_map, yrn)
            character_move.own_charcter_move(now_target_path)
            break
        elif yrn == back_button:
            cache_contorl.now_map = []
            cache_contorl.now_flow_id = "in_scene"
            break
        elif yrn in move_path_cmd:
            move_list_id = move_path_cmd.index(yrn)
            move_id = move_path_list[move_list_id]
            now_target_path = map_handle.get_scene_path_for_map_scene_id(
                now_map, move_id
            )
            character_move.own_charcter_move(now_target_path)
            break
        elif up_map_button != "Null" and yrn == up_map_button:
            up_map_path = map_handle.get_map_for_path(now_map)
            cache_contorl.now_map = up_map_path
        elif down_map_button != "Null" and yrn == down_map_button:
            character_position = cache_contorl.character_data[0].position
            down_map_scene_id = map_handle.get_map_scene_id_for_scene_path(
                cache_contorl.now_map, character_position
            )
            now_map.append(down_map_scene_id)
            cache_contorl.now_map = now_map
        elif yrn == show_scene_name_list_cmd:
            panel_state_handle.panel_state_change(show_scene_name_list_cmd)
