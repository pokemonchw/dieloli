import math
from Script.Core import cache_contorl, game_init, py_cmd, game_config
from Script.Design import (
    map_handle,
    panel_state_handle,
    handle_instruct,
    talk_cache,
    talk,
)
from Script.Panel import (
    in_scene_panel,
    see_character_attr_panel,
    instruct_panel,
)


def get_in_scene_func():
    """
    用于进入场景界面的流程
    """
    py_cmd.clr_cmd()
    cache_contorl.character_data[0].behavior["StartTime"] = cache_contorl.game_time
    scene_path = cache_contorl.character_data[0].position
    scene_path_str = map_handle.get_map_system_path_str_for_list(scene_path)
    map_handle.sort_scene_character_id(scene_path_str)
    cache_contorl.now_map = map_handle.get_map_for_path(scene_path)
    scene_data = cache_contorl.scene_data[scene_path_str].copy()
    scene_character_list = scene_data["SceneCharacterData"]
    if 0 not in scene_character_list:
        character_id_list = [0]
        scene_character_list = scene_character_list + character_id_list
        cache_contorl.scene_data[scene_path_str][
            "SceneCharacterData"
        ] = scene_character_list
    if (
        len(scene_character_list) > 1
        and not cache_contorl.character_data[0].target_character_id
    ):
        now_name_list = map_handle.get_scene_character_name_list(scene_path_str)
        now_name_list.remove(cache_contorl.character_data[0].name)
        cache_contorl.character_data[
            0
        ].target_character_id = map_handle.get_character_id_by_character_name(
            now_name_list[0], scene_path_str
        )
        if cache_contorl.old_character_id != 0:
            cache_contorl.character_data[
                0
            ].target_character_id = cache_contorl.old_character_id
            cache_contorl.old_character_id = 0
    if len(scene_character_list) > 1:
        see_scene_func(True)
    else:
        see_scene_func(False)


def see_scene_func(judge: bool):
    """
    用于查看当前场景界面的流程
    Keyword argument:
    judge -- 判断是否绘制角色列表界面的开关
    """
    while True:
        talk_cache.me = cache_contorl.character_data[0]
        talk_cache.tg = cache_contorl.character_data[
            cache_contorl.character_data[0].target_character_id
        ]
        input_s = []
        in_scene_panel.see_scene_panel()
        scene_path = cache_contorl.character_data[0].position
        scene_path_str = map_handle.get_map_system_path_str_for_list(scene_path)
        scene_character_name_list = map_handle.get_scene_character_name_list(
            scene_path_str
        )
        name_list_max = int(game_config.in_scene_see_player_max)
        change_page_judge = False
        if len(scene_character_name_list) == 1:
            cache_contorl.character_data[0].target_character_id = 0
        in_scene_cmd_list_1 = []
        now_start_id = len(instruct_panel.instruct_text_data)
        if judge:
            if cache_contorl.panel_state["SeeSceneCharacterListPage"] == "0":
                input_s = input_s + in_scene_panel.see_scene_character_list_panel()
                if len(scene_character_name_list) > name_list_max:
                    in_scene_cmd_list_1 = (
                        in_scene_panel.change_scene_character_list_panel()
                    )
                    change_page_judge = True
            input_s.append("SeeSceneCharacterListPage")
        start_id_1 = len(in_scene_cmd_list_1) + now_start_id
        in_scene_panel.see_character_info_panel()
        see_character_attr_panel.see_character_hp_and_mp_in_sence(
            cache_contorl.character_data[0].target_character_id
        )
        see_character_attr_panel.see_character_status_panel(
            cache_contorl.character_data[0].target_character_id
        )
        instruct_head = instruct_panel.see_instruct_head_panel()
        instruct_cmd = instruct_panel.instract_list_panel()
        in_scene_cmd_list_2 = in_scene_panel.in_scene_button_panel(start_id_1)
        if change_page_judge:
            input_s += in_scene_cmd_list_1 + instruct_head + in_scene_cmd_list_2
        else:
            input_s += instruct_head + in_scene_cmd_list_2
        input_s += instruct_cmd
        yrn = game_init.askfor_all(input_s)
        py_cmd.clr_cmd()
        now_page = int(cache_contorl.panel_state["SeeSceneCharacterListPanel"])
        character_max = len(cache_contorl.character_data) - 1
        page_max = math.floor(character_max / name_list_max)
        if yrn in scene_character_name_list:
            cache_contorl.character_data[
                0
            ].target_character_id = map_handle.get_character_id_by_character_name(
                yrn, scene_path_str
            )
        elif yrn in instruct_cmd:
            handle_instruct.handle_instruct(
                instruct_panel.instruct_id_cmd_data[int(yrn)]
            )
        elif (
            judge
            and yrn not in in_scene_cmd_list_2
            and yrn != "SeeSceneCharacterListPage"
            and change_page_judge
        ):
            if yrn == in_scene_cmd_list_1[0]:
                cache_contorl.panel_state["SeeSceneCharacterListPanel"] = 0
            elif yrn == in_scene_cmd_list_1[1]:
                if int(now_page) == 0:
                    cache_contorl.panel_state["SeeSceneCharacterListPanel"] = page_max
                else:
                    cache_contorl.panel_state["SeeSceneCharacterListPanel"] = (
                        int(now_page) - 1
                    )
            elif yrn == in_scene_cmd_list_1[2]:
                cache_contorl.panel_state[
                    "SeeSceneCharacterListPanel"
                ] = in_scene_panel.jump_character_list_page_panel()
            elif yrn == in_scene_cmd_list_1[3]:
                if int(now_page) == page_max:
                    cache_contorl.panel_state["SeeSceneCharacterListPanel"] = 0
                else:
                    cache_contorl.panel_state["SeeSceneCharacterListPanel"] = (
                        int(now_page) + 1
                    )
            elif yrn == in_scene_cmd_list_1[4]:
                cache_contorl.panel_state["SeeSceneCharacterListPanel"] = page_max
        elif yrn == in_scene_cmd_list_2[0]:
            cache_contorl.now_flow_id = "see_map"
            now_map = map_handle.get_map_for_path(
                cache_contorl.character_data[0].position
            )
            cache_contorl.now_map = now_map
            break
        elif yrn in [in_scene_cmd_list_2[1], in_scene_cmd_list_2[2]]:
            if yrn == in_scene_cmd_list_2[2]:
                cache_contorl.old_character_id = cache_contorl.character_data[
                    0
                ].target_character_id
                cache_contorl.character_data[0].target_character_id = 0
            cache_contorl.now_flow_id = "see_character_attr"
            cache_contorl.old_flow_id = "in_scene"
            break
        elif yrn == "SeeSceneCharacterListPage":
            panel_state_handle.panel_state_change(yrn)
        elif yrn in instruct_head:
            if cache_contorl.instruct_filter[yrn] == 0:
                cache_contorl.instruct_filter[yrn] = 1
            else:
                cache_contorl.instruct_filter[yrn] = 0
