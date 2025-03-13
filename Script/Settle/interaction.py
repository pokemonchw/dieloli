from types import FunctionType
from Script.Design import (
    settle_behavior,
    map_handle,
    cooking,
    game_time,
    character,
    constant,
    map_handle
)
from Script.Core import (
    get_text,
    game_type,
    cache_control,
)
from Script.Config import game_config, normal_config
from Script.UI.Model import draw

_: FunctionType = get_text._
""" 翻译api """
window_width: int = normal_config.config_normal.text_width
""" 窗体宽度 """
cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.LET_TARGET_CHAT_SELF)
def handle_let_target_chat_self(
        character_id: int,
        add_time: int,
        change_data: game_type.CharacterStatusChange,
        now_time: int,
):
    """
    让交互对象和自己聊天
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    if character_data.target_character_id == -1:
        return
    target_character_id = character_data.target_character_id
    target_character_data = cache.character_data[target_character_id]
    if target_character_data.target_character_id == character_id:
        if target_character_data.state == constant.CharacterStatus.STATUS_CHAT:
            return
    if target_character_data.state == constant.CharacterStatus.STATUS_MOVE:
        target_character_data.state = constant.CharacterStatus.STATUS_ARDER
        target_character_data.behavior.behavior_id = constant.Behavior.SHARE_BLANKLY
        target_character_data.behavior.move_src = []
        target_character_data.behavior.move_target = []
    if target_character_data.state != constant.CharacterStatus.STATUS_ARDER:
        now_add_time = 1
        if now_time > target_character_data.behavior.start_time:
            now_add_time = now_time - target_character_data.behavior.start_time
        constant.settle_behavior_effect_data[constant.BehaviorEffect.INTERRUPT_TARGET_ACTIVITY](character_id, now_add_time, None, now_time)
    target_character_data.target_character_id = character_id
    target_character_data.state = constant.CharacterStatus.STATUS_CHAT
    target_character_data.behavior.behavior_id = constant.Behavior.CHAT
    target_character_data.behavior.start_time = now_time
    target_character_data.behavior.duration = character_data.behavior.duration


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.LET_ALL_STUDENTS_STUDY_IN_CLASSROOM)
def handle_let_all_students_study_in_classroom(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    让教室内所有本班学生进入学习状态
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    if 1 not in character_data.identity_data:
        return
    identity_data: game_type.TeacherIdentity = character_data.identity_data[1]
    if identity_data.now_classroom == "":
        return
    classroom_path_str = identity_data.now_classroom
    classroom_path = map_handle.get_map_system_path_for_str(classroom_path_str)
    if classroom_path != character_data.position:
        return
    old_target_character_id = character_data.target_character_id
    scene_data: game_type.Scene = cache.scene_data[classroom_path_str]
    for now_character_id in scene_data.character_list:
        if now_character_id == character_id:
            continue
        now_character_data: game_type.Character = cache.character_data[now_character_id]
        if 0 not in now_character_data.identity_data:
            continue
        now_identity_data: game_type.StudentIdentity = now_character_data.identity_data[0]
        if now_identity_data.classroom != classroom_path_str:
            continue
        if now_character_data.state == constant.CharacterStatus.STATUS_ATTEND_CLASS:
            continue
        if now_character_data.state != constant.CharacterStatus.STATUS_ARDER:
            now_add_time = 1
            if now_time > now_character_data.behavior.start_time:
                now_add_time = now_time - now_character_data.behavior.start_time
            character_data.target_character_id = now_character_id
            constant.settle_behavior_effect_data[constant.BehaviorEffect.INTERRUPT_TARGET_ACTIVITY](character_id, now_add_time, None, now_time)
        now_character_data.behavior.start_time = now_time
        constant.handle_state_machine_data[constant.StateMachine.ATTEND_CLASS](now_character_id)
    character_data.target_character_id = old_target_character_id


@settle_behavior.add_settle_behavior_effect(constant.BehaviorEffect.LET_ALL_NOT_STUDENTS_LEAVING_CLASSROOM)
def handle_let_all_not_students_leaving_classroom(
    character_id: int,
    add_time: int,
    change_data: game_type.CharacterStatusChange,
    now_time: int,
):
    """
    让教室内所有非本班学生离开教室
    Keyword arguments:
    character_id -- 角色id
    add_time -- 结算时间
    change_data -- 状态变更信息记录对象
    now_time -- 结算的时间戳
    """
    character_data: game_type.Character = cache.character_data[character_id]
    if character_data.dead:
        return
    if 1 not in character_data.identity_data:
        return
    identity_data: game_type.TeacherIdentity = character_data.identity_data[1]
    if identity_data.now_classroom == "":
        return
    classroom_path_str = identity_data.now_classroom
    classroom_path = map_handle.get_map_system_path_for_str(classroom_path_str)
    if classroom_path != character_data.position:
        return
    old_target_character_id = character_data.target_character_id
    scene_data: game_type.Scene = cache.scene_data[classroom_path_str]
    for now_character_id in list(scene_data.character_list):
        if now_character_id == character_id:
            continue
        now_judge = False
        now_character_data: game_type.Character = cache.character_data[now_character_id]
        if now_character_data.state == constant.CharacterStatus.STATUS_MOVE:
            continue
        if 0 not in now_character_data.identity_data:
            now_judge = True
        if not now_judge:
            now_identity_data: game_type.StudentIdentity = now_character_data.identity_data[0]
            if now_identity_data.classroom != classroom_path_str:
                now_judge = True
        if now_judge:
            now_add_time = 1
            if now_time > now_character_data.behavior.start_time:
                now_add_time = now_time - now_character_data.behavior.start_time
            character_data.target_character_id = now_character_id
            if now_character_data.state != constant.CharacterStatus.STATUS_ARDER:
                constant.settle_behavior_effect_data[constant.BehaviorEffect.INTERRUPT_TARGET_ACTIVITY](character_id, now_add_time, None, now_time)
            now_character_data.behavior.start_time = now_time
            constant.handle_state_machine_data[constant.StateMachine.MOVE_TO_NEAREST_NOT_CLASSROOM](now_character_id)
            if not now_character_id:
                now_draw = draw.LeftDraw()
                now_draw.width = window_width
                now_draw.text = _("{NickName}被赶出了教室").format(NickName=character_data.nick_name)
                now_draw.draw()
    character_data.target_character_id = old_target_character_id


