from Script.Core import cache_contorl, text_loading, era_print, constant
from Script.Design import map_handle, game_time, update


def own_charcter_move(target_scene: list):
    """
    主角寻路至目标场景
    Keyword arguments:
    target_scene -- 寻路目标场景(在地图系统下的绝对坐标)
    """
    (
        move_now,
        now_path_list,
        now_target_position,
        now_need_time,
    ) = character_move(0, target_scene)
    if move_now == "Null":
        null_message = text_loading.get_text_data(
            constant.FilePath.MESSAGE_PATH, "30"
        )
        era_print.normal_print(null_message)
    else:
        map_handle.character_move_scene(
            cache_contorl.character_data[0].position, now_target_position, 0,
        )
        game_time.sub_time_now(now_need_time)
        if cache_contorl.character_data[0].position != target_scene:
            own_charcter_move(target_scene)
    update.game_update_flow()
    cache_contorl.now_character_id = 0
    cache_contorl.now_flow_id = "in_scene"


def character_move(
    character_id: str, target_scene: list
) -> (str, list, list, int):
    """
    通用角色移动控制
    Keyword arguments:
    character_id -- 角色id
    target_scene -- 寻路目标场景(在地图系统下的绝对坐标)
    Return arguments:
    str:null -- 未找到路径
    str:end -- 当前位置已是路径终点
    list -- 路径
    list -- 本次移动到的位置
    int -- 本次移动花费的时间
    """
    now_position = cache_contorl.character_data[character_id].position
    scene_hierarchy = map_handle.judge_scene_affiliation(
        now_position, target_scene
    )
    if scene_hierarchy == "common":
        map_path = map_handle.get_common_map_for_scene_path(
            now_position, target_scene
        )
        now_map_scene_id = map_handle.get_map_scene_id_for_scene_path(
            map_path, now_position
        )
        target_map_scene_id = map_handle.get_map_scene_id_for_scene_path(
            map_path, target_scene
        )
        return identical_map_move(
            character_id, map_path, now_map_scene_id, target_map_scene_id
        )
    return difference_map_move(character_id, target_scene)


def difference_map_move(
    character_id: str, target_scene: list
) -> (str, list, list, int):
    """
    角色跨地图层级移动
    Keyword arguments:
    character_id -- 角色id
    target_scene -- 寻路目标场景(在地图系统下的绝对坐标)
    Return arguments:
    str:null -- 未找到路径
    str:end -- 当前位置已是路径终点
    list -- 路径
    list -- 本次移动到的位置
    int -- 本次移动花费的时间
    """
    now_position = cache_contorl.character_data[character_id].position
    is_affiliation = map_handle.judge_scene_is_affiliation(
        now_position, target_scene
    )
    now_true_position = map_handle.get_scene_path_for_true(now_position)
    map_door_data = map_handle.get_map_door_data_for_scene_path(
        map_handle.get_map_system_path_str_for_list(now_true_position)
    )
    door_scene = "0"
    now_true_map = map_handle.get_map_for_path(now_true_position)
    now_true_map_map_system_str = map_handle.get_map_system_path_str_for_list(
        now_true_map
    )
    if is_affiliation == "subordinate":
        now_true_affiliation = map_handle.judge_scene_is_affiliation(
            now_true_position, target_scene
        )
        if now_true_affiliation == "subordinate":
            if map_door_data != {}:
                door_scene = map_door_data[now_true_map_map_system_str]["Door"]
            now_map_scene_id = map_handle.get_map_scene_id_for_scene_path(
                now_true_map, now_position
            )
            return identical_map_move(
                character_id, now_true_map, now_map_scene_id, door_scene
            )
        now_map = map_handle.get_map_for_path(target_scene)
        now_map_scene_id = map_handle.get_map_scene_id_for_scene_path(
            now_map, now_position
        )
        return identical_map_move(
            character_id, now_map, now_map_scene_id, door_scene
        )
    else:
        if now_true_map == []:
            now_target_map_scene_id = map_handle.get_map_scene_id_for_scene_path(
                [], target_scene
            )
            now_map_scene_id = map_handle.get_map_scene_id_for_scene_path(
                [], now_true_position
            )
            return identical_map_move(
                character_id, [], now_map_scene_id, now_target_map_scene_id
            )
        relation_map_list = map_handle.get_relation_map_list_for_scene_path(
            now_true_position
        )
        now_scene_real_map = relation_map_list[-1]
        common_map = map_handle.get_common_map_for_scene_path(
            now_true_position, target_scene
        )
        real_map_in_map = map_handle.get_map_for_path(now_scene_real_map)
        target_map_scene_id = map_handle.get_map_scene_id_for_scene_path(
            common_map, target_scene
        )
        if now_scene_real_map == common_map:
            now_map_scene_id = map_handle.get_map_scene_id_for_scene_path(
                common_map, now_true_position
            )
        elif real_map_in_map == common_map:
            now_map_scene_id = map_handle.get_map_scene_id_for_scene_path(
                common_map, now_scene_real_map
            )
        else:
            now_map_scene_id = map_handle.get_map_scene_id_for_scene_path(
                now_true_map, now_true_position
            )
            target_map_scene_id = "0"
            common_map = now_scene_real_map
        return identical_map_move(
            character_id, common_map, now_map_scene_id, target_map_scene_id
        )


def identical_map_move(
    character_id: str,
    now_map: list,
    now_map_scene_id: str,
    target_map_scene_id: str,
) -> (str, list, list, int):
    """
    角色在相同地图层级内移动
    Keyword arguments:
    character_id -- 角色id
    now_map -- 当前地图路径
    now_map_scene_id -- 当前角色所在场景(当前地图层级下的相对坐标)
    target_map_scene_id -- 寻路目标场景(当前地图层级下的相对坐标)
    Return arguments:
    str:null -- 未找到路径
    str:end -- 当前位置已是路径终点
    list -- 路径
    list -- 本次移动到的位置
    int -- 本次移动花费的时间
    """
    now_map_str = map_handle.get_map_system_path_str_for_list(now_map)
    move_end, move_path = map_handle.get_path_finding(
        now_map_str, now_map_scene_id, target_map_scene_id
    )
    now_target_position = []
    now_need_time = 0
    if move_path != []:
        now_target_scene_id = move_path["Path"][0]
        now_need_time = move_path["Time"][0]
        now_character_position = map_handle.get_scene_path_for_map_scene_id(
            now_map, now_map_scene_id
        )
        now_target_position = map_handle.get_scene_path_for_map_scene_id(
            now_map, now_target_scene_id
        )
    return move_end, move_path, now_target_position, now_need_time
