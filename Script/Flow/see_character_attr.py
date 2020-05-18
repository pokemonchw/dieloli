from Script.Core import (
    era_print,
    cache_contorl,
    py_cmd,
    game_init,
    text_loading,
    game_config,
    constant,
)
from Script.Design import (
    panel_state_handle,
    map_handle,
    character_handle,
)
from Script.Panel import see_character_attr_panel
import math
from Script.Flow import game_start_flow

panel_list = [
    "CharacterMainAttrPanel",
    "CharacterEquipmentPanel",
    "CharacterItemPanel",
    "CharacterExperiencePanel",
    "CharacterLevelPanel",
    "CharacterFeaturesPanel",
    "CharacterEngravingPanel",
]


def acknowledgment_attribute_func():
    """
    创建角色时用于查看角色属性的流程
    """
    while True:
        character_handle.init_character_list()
        input_s = []
        see_attr_in_every_time_func()
        flow_return = see_character_attr_panel.input_attr_over_panel()
        input_s = flow_return + input_s
        yrn = game_init.askfor_int(input_s)
        py_cmd.clr_cmd()
        if yrn in panel_list:
            panel_state_handle.panel_state_change(yrn)
        elif yrn == "0":
            game_start_flow.init_game_start()
            break
        elif yrn == "1":
            cache_contorl.wframe_mouse.w_frame_re_print = 1
            era_print.next_screen_print()
            cache_contorl.now_flow_id = "title_frame"
            break


def see_attr_on_every_time_func():
    """
    通用用于查看角色属性的流程
    """
    while True:
        character_id = cache_contorl.now_character_id
        if cache_contorl.old_flow_id == "in_scene":
            now_scene = cache_contorl.character_data[0].position
            now_scene_str = map_handle.get_map_system_path_str_for_list(
                now_scene
            )
            character_id_list = map_handle.get_scene_character_id_list(
                now_scene_str
            )
        else:
            character_id_list = list(cache_contorl.character_data.keys())
        character_id_index = character_id_list.index(character_id)
        input_s = []
        see_attr_in_every_time_func()
        ask_see_attr = see_character_attr_panel.ask_for_see_attr()
        input_s += ask_see_attr
        inputs_1 = see_character_attr_panel.ask_for_see_attr_cmd()
        input_s += inputs_1
        yrn = game_init.askfor_all(input_s)
        py_cmd.clr_cmd()
        show_attr_handle_data = text_loading.get_text_data(
            constant.FilePath.CMD_PATH, "seeAttrPanelHandle"
        )
        character_max = character_id_list[len(character_id_list) - 1]
        if yrn in show_attr_handle_data:
            cache_contorl.panel_state["AttrShowHandlePanel"] = yrn
        elif yrn == "0":
            if character_id_index == 0:
                cache_contorl.now_character_id = character_max
            else:
                character_id = character_id_list[character_id_index - 1]
                cache_contorl.now_character_id = character_id
        elif yrn == "1":
            if cache_contorl.old_flow_id == "main":
                cache_contorl.now_character_id = 0
            elif cache_contorl.old_flow_id == "see_character_list":
                character_list_show = int(game_config.character_list_show)
                now_page_id = character_id_index / character_list_show
                cache_contorl.panel_state[
                    "SeeCharacterListPanel"
                ] = now_page_id
            elif cache_contorl.old_flow_id == "in_scene":
                scene_path = cache_contorl.character_data[0].position
                scene_path_str = map_handle.get_map_system_path_str_for_list(
                    scene_path
                )
                name_list = map_handle.get_scene_character_name_list(
                    scene_path_str, True
                )
                now_character_name = cache_contorl.character_data[
                    cache_contorl.now_character_id
                ].name
                try:
                    now_character_index = name_list.index(now_character_name)
                except ValueError:
                    now_character_index = 0
                name_list_max = int(game_config.in_scene_see_player_max)
                now_scene_character_list_page = math.floor(
                    now_character_index / name_list_max
                )
                cache_contorl.panel_state[
                    "SeeSceneCharacterListPanel"
                ] = now_scene_character_list_page
            cache_contorl.panel_state["AttrShowHandlePanel"] = "MainAttr"
            cache_contorl.now_flow_id = cache_contorl.old_flow_id
            cache_contorl.old_flow_id = cache_contorl.too_old_flow_id
            break
        elif yrn == "2":
            if character_id == character_max:
                character_id = character_id_list[0]
                cache_contorl.now_character_id = character_id
            else:
                character_id = character_id_list[character_id_index + 1]
                cache_contorl.now_character_id = character_id


def see_attr_in_every_time_func():
    """
    用于在任何时候查看角色属性的流程
    """
    character_id = cache_contorl.now_character_id
    now_attr_panel = cache_contorl.panel_state["AttrShowHandlePanel"]
    return see_character_attr_panel.panel_data[now_attr_panel](character_id)
