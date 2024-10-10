from types import FunctionType
from Script.Core import cache_control, game_type, get_text, flow_handle
from Script.Design import map_handle, update, constant
from Script.UI.Moudle import panel

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
_: FunctionType = get_text._
""" 翻译api """


def own_charcter_move(target_scene: list):
    """
    主角寻路至目标场景
    Keyword arguments:
    target_scene -- 寻路目标场景(在地图系统下的绝对坐标)
    """
    reset_target_judge = True
    while 1:
        character_data: game_type.Character = cache.character_data[0]
        if character_data.position != target_scene:
            (
                move_now,
                _,
                now_target_position,
                now_need_time,
            ) = character_move(0, target_scene)
            if move_now == "Null":
                break
            character_data.behavior.behavior_id = constant.Behavior.MOVE
            character_data.behavior.move_target = now_target_position
            character_data.behavior.duration = now_need_time
            character_data.state = constant.CharacterStatus.STATUS_MOVE
            update.game_update_flow(now_need_time)
            if character_data.position != target_scene:
                if ask_own_black_move():
                    reset_target_judge = False
                    break
        else:
            break
    if reset_target_judge:
        cache.character_data[0].target_character_id = -1
    cache.now_panel_id = constant.Panel.IN_SCENE


def ask_own_black_move() -> bool:
    """
    在玩家向目标场景移动时，在还未抵达目标场景时，若场景中存在被玩家收藏的角色，则询问是否需要停止移动
    Return arguments:
    bool -- 是否结束移动循环
    """
    character_data: game_type.Character = cache.character_data[0]
    if len(character_data.collection_character) == 0:
        return False
    now_scene_path = character_data.position
    now_scene_path_str = map_handle.get_map_system_path_str_for_list(now_scene_path)
    now_scene_data: game_type.Scene = cache.scene_data[now_scene_path_str]
    if len(now_scene_data.character_list) == 1:
        return False
    collection_character = now_scene_data.character_list & character_data.collection_character
    if len(collection_character) == 0:
        return False
    target_name_list = []
    target_character_id = -1
    for now_target_character_id in collection_character:
        if target_character_id == -1:
            target_character_id = now_target_character_id
        target_character: game_type.Character = cache.character_data[now_target_character_id]
        target_name_list.append(target_character.name)
    ask_draw = panel.OneMessageAndSingleColumnButton()
    ask_draw.set([_("是"), _("否")],_("在这里遇到了{CharacterList}，要停下来吗？").format(CharacterList=target_name_list), 0)
    ask_list = ask_draw.get_return_list()
    ask_draw.draw()
    yrn = flow_handle.askfor_all(ask_list)
    if yrn == "0":
        character_data.target_character_id = target_character_id
        return True
    return False



def own_move_to_character_scene(target_character_id: int):
    """
    寻路到目标对象所在场景
    Keyword arguments:
    target_character_id -- 目标角色id
    """
    while 1:
        character_data: game_type.Character = cache.character_data[0]
        target_data: game_type.Character = cache.character_data[target_character_id]
        if character_data.position != target_data.position:
            (move_now, _, now_target_position, now_need_time) = character_move(0, target_data.position)
            if move_now == "Null":
                break
            character_data.behavior.behavior_id = constant.Behavior.MOVE
            character_data.behavior.move_target = now_target_position
            character_data.behavior.duration = now_need_time
            character_data.state = constant.CharacterStatus.STATUS_MOVE
            update.game_update_flow(now_need_time)
        else:
            break
    cache.character_data[0].target_character_id = target_character_id
    cache.now_panel_id = constant.Panel.IN_SCENE


def character_move(character_id: int, target_scene: list) -> (str, list, list, int):
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
    if now_position == target_scene:
        return "end", [], [], 0
    now_position_str = map_handle.get_map_system_path_str_for_list(now_position)
    target_scene_str = map_handle.get_map_system_path_str_for_list(target_scene)
    if (
        now_position_str not in map_handle.scene_path_edge
        or target_scene_str not in map_handle.scene_path_edge[now_position_str]
    ):
        return "null", [], [], 0
    now_path_data = map_handle.scene_path_edge[now_position_str][target_scene_str]
    return "", [], now_path_data[0], now_path_data[1]
