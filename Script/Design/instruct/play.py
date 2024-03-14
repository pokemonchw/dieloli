from types import FunctionType
from Script.Core import cache_control, game_type, get_text
from Script.Design import update, character, constant, handle_instruct
from Script.Config import normal_config


cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
_: FunctionType = get_text._
""" 翻译api """
width: int = normal_config.config_normal.text_width
""" 屏幕宽度 """


@handle_instruct.add_instruct(constant.Instruct.PLAY_COMPUTER, constant.InstructType.PLAY, _("玩电脑"), {constant.Premise.IN_COMPUTER_ROOM_SCENE})
def handle_play_computer():
    """ 处理玩电脑指令 """
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data = cache.character_data[0]
    character_data.behavior.duration = 30
    character_data.behavior.behavior_id = constant.Behavior.PLAY_COMPUTER
    character_data.state = constant.CharacterStatus.STATUS_PLAY_COMPUTER
    update.game_update_flow(30)


@handle_instruct.add_instruct(constant.Instruct.DRAW, constant.InstructType.PLAY, _("画画"), {})
def handle_draw():
    """ 处理画画指令 """
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data = cache.character_data[0]
    character_data.behavior.duration = 30
    character_data.behavior.behavior_id = constant.Behavior.DRAW
    character_data.state = constant.CharacterStatus.STATUS_DRAW
    update.game_update_flow(30)
