from types import FunctionType
import time
from Script.Core import cache_control, game_type, get_text
from Script.Design import update, character, constant, handle_instruct
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
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.CHAT
    character_data.state = constant.CharacterStatus.STATUS_CHAT
    update.game_update_flow(10)


@handle_instruct.add_instruct(constant.Instruct.ABUSE, constant.InstructType.DIALOGUE,_("辱骂"),{constant.Premise.HAVE_TARGET})
def handle_abuse():
    """处理辱骂指令"""
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data = cache.character_data[0]
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.ABUSE
    character_data.state = constant.CharacterStatus.STATUS_ABUSE
    update.game_update_flow(10)


@handle_instruct.add_instruct(constant.Instruct.GENERAL_SPEECH, constant.InstructType.DIALOGUE,_("演讲"),{})
def handle_general_speech():
    """ 处理演讲指令 """
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data = cache.character_data[0]
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.GENERAL_SPEECH
    character_data.state = constant.CharacterStatus.STATUS_GENERAL_SPEECH
    update.game_update_flow(10)
