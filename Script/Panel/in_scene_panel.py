from Script.Core import (
    cache_contorl,
    text_loading,
    era_print,
    py_cmd,
    game_config,
    constant,
)
from Script.Design import (
    game_time,
    cmd_button_queue,
    map_handle,
    character_handle,
    input_queue,
    attr_text,
)
import math

panel_state_text_data = text_loading.get_text_data(
    constant.FilePath.CMD_PATH, "cmdSwitch"
)
panel_state_on_text = panel_state_text_data[1]
panel_state_off_text = panel_state_text_data[0]


def see_scene_panel():
    """
    当前场景信息面板
    """
    title_text = text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "75")
    era_print.little_title_print(title_text)
    time_text = game_time.get_date_text()
    era_print.normal_print(time_text)
    era_print.normal_print(" ")
    scene_path = cache_contorl.character_data[0].position
    scene_path_str = map_handle.get_map_system_path_str_for_list(scene_path)
    map_list = map_handle.get_map_hierarchy_list_for_scene_path(scene_path, [])
    map_path_text = ""
    map_list.reverse()
    for now_map in map_list:
        now_map_map_system_str = map_handle.get_map_system_path_str_for_list(now_map)
        map_name = cache_contorl.map_data[now_map_map_system_str]["MapName"]
        map_path_text += map_name + "-"
    scene_data = cache_contorl.scene_data[scene_path_str]
    scene_name = map_path_text + scene_data["SceneName"]
    scene_info_head = text_loading.get_text_data(
        constant.FilePath.STAGE_WORD_PATH, "76"
    )
    scene_info = scene_info_head + scene_name
    era_print.normal_print(scene_info)
    panel_state = cache_contorl.panel_state["SeeSceneCharacterListPage"]
    switch = panel_state_on_text
    if panel_state == "0":
        switch = panel_state_off_text
    scene_character_list = scene_data["SceneCharacterData"]
    if len(scene_character_list) > 1:
        era_print.normal_print(" ")
        py_cmd.pcmd(switch, "SeeSceneCharacterListPage")
    era_print.little_line_print()


def see_scene_character_list_panel() -> list:
    """
    当前场景角色列表面板
    """
    input_s = []
    scene_path = cache_contorl.character_data[0].position
    scene_path_str = map_handle.get_map_system_path_str_for_list(scene_path)
    name_list = map_handle.get_scene_character_name_list(scene_path_str, True)
    name_list = get_now_page_name_list(name_list)
    if len(name_list) > 0:
        see_character_text = text_loading.get_text_data(
            constant.FilePath.MESSAGE_PATH, "26"
        )
        era_print.normal_print(see_character_text)
        era_print.line_feed_print()
        character_id = cache_contorl.character_data[0].target_character_id
        character_data = cache_contorl.character_data[character_id]
        character_name = character_data.name
        input_s = cmd_button_queue.option_str(
            "",
            cmd_column=10,
            cmd_size="center",
            askfor=False,
            cmd_list_data=name_list,
            null_cmd=character_name,
        )
    return input_s


def change_scene_character_list_panel(start_id: int) -> list:
    """
    当前场景角色列表页切换控制面板
    Keyword arguments:
    start_id -- 指令的起始id
    """
    name_list_max = int(game_config.in_scene_see_player_max)
    now_page = int(cache_contorl.panel_state["SeeSceneCharacterListPanel"])
    scene_path = cache_contorl.character_data[0].position
    scene_path_str = map_handle.get_map_system_path_str_for_list(scene_path)
    scene_character_name_list = map_handle.get_scene_character_name_list(scene_path_str)
    character_max = len(scene_character_name_list)
    page_max = math.floor(character_max / name_list_max)
    page_text = "(" + str(now_page) + "/" + str(page_max) + ")"
    input_s = cmd_button_queue.option_int(
        constant.CmdMenu.CHANGE_SCENE_CHARACTER_LIST,
        cmd_column=5,
        askfor=False,
        cmd_size="center",
        start_id=start_id,
    )
    era_print.page_line_print(sample="-", string=page_text)
    era_print.line_feed_print()
    return input_s


def get_now_page_name_list(name_list: list) -> list:
    """
    获取当前角色列表页面角色姓名列表
    Keyword arguments:
    name_list -- 当前场景下角色列表
    """
    now_page = int(cache_contorl.panel_state["SeeSceneCharacterListPanel"])
    name_list_max = int(game_config.in_scene_see_player_max)
    new_name_list = []
    now_name_start_id = now_page * name_list_max
    for i in range(now_name_start_id, now_name_start_id + name_list_max):
        if i < len(name_list):
            new_name_list.append(name_list[i])
    return new_name_list


def see_character_info_panel():
    """
    查看当前互动对象信息面板
    """
    character_info = text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "77")
    era_print.normal_print(character_info)
    character_id = cache_contorl.character_data[0].target_character_id
    character_data = cache_contorl.character_data[character_id]
    character_name = character_data.name
    era_print.normal_print(character_name)
    era_print.normal_print(" ")
    sex = character_data.sex
    sex_text = text_loading.get_text_data(
        constant.FilePath.STAGE_WORD_PATH, "2"
    ) + attr_text.get_sex_text(sex)
    era_print.normal_print(sex_text)
    era_print.normal_print(" ")
    intimate_info = text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "16")
    graces_info = text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "17")
    character_intimate_text = intimate_info + f"{character_data.intimate}"
    character_graces_text = graces_info + f"{character_data.graces}"
    era_print.normal_print(character_intimate_text)
    era_print.normal_print(" ")
    era_print.normal_print(character_graces_text)
    state_text = attr_text.get_state_text(character_id)
    era_print.normal_print(" ")
    era_print.normal_print(state_text)
    era_print.little_line_print()


def jump_character_list_page_panel() -> str:
    """
    角色列表页面跳转控制面板
    """
    message_text = text_loading.get_text_data(constant.FilePath.MESSAGE_PATH, "32")
    name_list_max = int(game_config.in_scene_see_player_max)
    character_max = character_handle.get_character_index_max()
    page_max = math.floor(character_max / name_list_max)
    era_print.normal_print("\n" + message_text + "(0-" + str(page_max) + ")")
    ans = input_queue.wait_input(0, page_max)
    era_print.normal_print(ans)
    return ans


def in_scene_button_panel(start_id: int) -> list:
    """
    场景页面基础控制菜单面板
    Keyword arguments:
    start_id -- 基础控制菜单命令起始Id
    """
    era_print.line_feed_print()
    era_print.restart_line_print(":")
    input_s = cmd_button_queue.option_int(
        cmd_list=constant.CmdMenu.IN_SCENE_LIST1,
        cmd_column=9,
        askfor=False,
        cmd_size="center",
        start_id=start_id,
    )
    return input_s
