import uuid
from types import FunctionType
import time
from Script.Core import cache_control, game_type, get_text
from Script.Design import update, character, constant, handle_instruct, map_handle, session_handler
from Script.Config import normal_config


cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
_: FunctionType = get_text._
""" 翻译api """
width: int = normal_config.config_normal.text_width
""" 屏幕宽度 """


@handle_instruct.add_instruct(
    constant.Instruct.CHAT, constant.InstructType.DIALOGUE, _("闲聊"), {constant.Premise.HAVE_TARGET}
)
def handle_chat():
    """处理闲聊指令"""
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data = cache.character_data[0]
    target_id = character_data.target_character_id
    
    # Send social request
    session_uid = str(uuid.uuid4())
    session = game_type.InteractionSession(0, [target_id], constant.Behavior.CHAT)
    session.uid = session_uid
    session.start_time = cache.game_time
    cache.interaction_sessions[session_uid] = session
    
    scene_path_str = map_handle.get_map_system_path_str_for_list(character_data.position)
    scene_data: game_type.Scene = cache.scene_data[scene_path_str]
    scene_data.social_fields[session_uid] = "Chat"
    
    handler = session_handler.get_session_handler(session_uid)
    if handler:
        handler.on_start()
    
    target_data = cache.character_data[target_id]
    target_data.social_requests.append({
        'initiator': 0,
        'session_uid': session_uid,
        'type': constant.Behavior.CHAT,
        'weight': 150 # Base weight
    })
    
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.CHAT
    character_data.state = constant.CharacterStatus.STATUS_SOCIAL_INTERACTING
    character_data.active_session = session_uid
    
    update.game_update_flow(10)


@handle_instruct.add_instruct(constant.Instruct.ABUSE, constant.InstructType.DIALOGUE,_("辱骂"),{constant.Premise.HAVE_TARGET})
def handle_abuse():
    """处理辱骂指令"""
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data = cache.character_data[0]
    target_id = character_data.target_character_id
    
    # Send social request
    session_uid = str(uuid.uuid4())
    session = game_type.InteractionSession(0, [target_id], constant.Behavior.ABUSE)
    session.uid = session_uid
    session.start_time = cache.game_time
    cache.interaction_sessions[session_uid] = session
    
    scene_path_str = map_handle.get_map_system_path_str_for_list(character_data.position)
    scene_data: game_type.Scene = cache.scene_data[scene_path_str]
    scene_data.social_fields[session_uid] = "Argument"
    
    handler = session_handler.get_session_handler(session_uid)
    if handler:
        handler.on_start()
    
    target_data = cache.character_data[target_id]
    target_data.social_requests.append({
        'initiator': 0,
        'session_uid': session_uid,
        'type': constant.Behavior.ABUSE,
        'weight': 400
    })
    
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.ABUSE
    character_data.state = constant.CharacterStatus.STATUS_SOCIAL_INTERACTING
    character_data.active_session = session_uid
    
    update.game_update_flow(10)


@handle_instruct.add_instruct(constant.Instruct.GENERAL_SPEECH, constant.InstructType.DIALOGUE,_("演讲"),set())
def handle_general_speech():
    """ 处理演讲指令 """
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data = cache.character_data[0]
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.GENERAL_SPEECH
    character_data.state = constant.CharacterStatus.STATUS_GENERAL_SPEECH
    update.game_update_flow(10)
