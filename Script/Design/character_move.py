from Script.Core import cache_control, constant, game_type
from Script.Design import map_handle, update, character

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


def own_charcter_move(target_scene: list):
    """
    主角寻路至目标场景
    Keyword arguments:
    target_scene -- 寻路目标场景(在地图系统下的绝对坐标)
    """
    while 1:
        character_data: game_type.Character = cache.character_data[0]
        if character_data.position != target_scene:
            (
                move_now,
                now_path_list,
                now_target_position,
                now_need_time,
            ) = character_move(0, target_scene)
            if move_now == "Null":
                break
            character_data.behavior.behavior_id = constant.Behavior.MOVE
            character_data.behavior.move_target = now_target_position
            character_data.behavior.duration = now_need_time
            character_data.state = constant.CharacterStatus.STATUS_MOVE
            character.init_character_behavior_start_time(0, cache.game_time)
            update.game_update_flow(now_need_time)
        else:
            break
    cache.character_data[0].target_character_id = 0
    cache.now_panel_id = constant.Panel.IN_SCENE


def character_move(character_id: str, target_scene: list) -> (str, list, list, int):
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
    now_position = cache.character_data[character_id].position
    scene_hierarchy = map_handle.judge_scene_affiliation(now_position, target_scene)
    if scene_hierarchy == "common":
        map_path = map_handle.get_common_map_for_scene_path(now_position, target_scene)
        now_map_scene_id = map_handle.get_map_scene_id_for_scene_path(map_path, now_position)
        target_map_scene_id = map_handle.get_map_scene_id_for_scene_path(map_path, target_scene)
        return identical_map_move(character_id, map_path, now_map_scene_id, target_map_scene_id)
    return difference_map_move(character_id, target_scene)


def difference_map_move(character_id: str, target_scene: list) -> (str, list, list, int):
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
    character_data = cache.character_data[character_id]
    now_position = character_data.position
    is_affiliation = map_handle.judge_scene_affiliation(now_position, target_scene)
    now_true_position = map_handle.get_scene_path_for_true(now_position)
    now_true_map = map_handle.get_map_for_path(now_true_position)
    if is_affiliation == "subordinate":
        now_true_affiliation = map_handle.judge_scene_is_affiliation(now_true_position, target_scene)
        if now_true_affiliation == "subordinate":
            now_map_scene_id = map_handle.get_map_scene_id_for_scene_path(now_true_map, now_true_position)
            return identical_map_move(character_id, now_true_map, now_map_scene_id, "0")
        now_map = map_handle.get_map_for_path(target_scene)
        now_map_scene_id = map_handle.get_map_scene_id_for_scene_path(now_map, now_position)
        return identical_map_move(character_id, now_map, now_map_scene_id, "0")
    relation_map_list = map_handle.get_relation_map_list_for_scene_path(now_true_position)
    now_scene_real_map = relation_map_list[-1]
    now_map_scene_id = map_handle.get_map_scene_id_for_scene_path(now_scene_real_map, now_true_position)
    common_map = map_handle.get_common_map_for_scene_path(now_true_position, target_scene)
    if now_scene_real_map != common_map:
        if now_map_scene_id == "0":
            now_true_position = now_scene_real_map.copy()
            relation_map_list = map_handle.get_relation_map_list_for_scene_path(now_true_position)
            now_scene_real_map = relation_map_list[-1]
    target_map_scene_id = map_handle.get_map_scene_id_for_scene_path(common_map, target_scene)
    if now_scene_real_map == common_map:
        now_map_scene_id = map_handle.get_map_scene_id_for_scene_path(common_map, now_true_position)
    else:
        now_map_scene_id = map_handle.get_map_scene_id_for_scene_path(now_scene_real_map, now_true_position)
        target_map_scene_id = "0"
        common_map = now_scene_real_map
    return identical_map_move(character_id, common_map, now_map_scene_id, target_map_scene_id)


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
    move_end, move_path = map_handle.get_path_finding(now_map_str, now_map_scene_id, target_map_scene_id)
    now_target_position = []
    now_need_time = 0
    if move_path != []:
        now_target_scene_id = move_path.path[0]
        now_need_time = move_path.time[0]
        now_target_position = map_handle.get_scene_path_for_map_scene_id(now_map, now_target_scene_id)
    return move_end, move_path, now_target_position, now_need_time
