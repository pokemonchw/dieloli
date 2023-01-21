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


@handle_instruct.add_instruct(constant.Instruct.SINGING, constant.InstructType.PERFORM, _("唱歌"), {})
def handle_singing():
    """处理唱歌指令"""
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data = cache.character_data[0]
    character_data.behavior.duration = 5
    character_data.behavior.behavior_id = constant.Behavior.SINGING
    character_data.state = constant.CharacterStatus.STATUS_SINGING
    update.game_update_flow(5)


@handle_instruct.add_instruct(constant.Instruct.DANCE,constant.InstructType.PERFORM,_("跳舞"),{})
def handle_dance():
    """处理跳舞指令"""
    character_data: game_type.Character = cache.character_data[0]
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data.behavior.duration = 5
    character_data.behavior.behavior_id = constant.Behavior.DANCE
    character_data.state = constant.CharacterStatus.STATUS_DANCE
    update.game_update_flow(5)


@handle_instruct.add_instruct(
    constant.Instruct.PLAY_PIANO,
    constant.InstructType.PERFORM,
    _("弹钢琴"),
    {constant.Premise.IN_MUSIC_CLASSROOM},
)
def handle_play_piano():
    """处理弹钢琴指令"""
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data = cache.character_data[0]
    character_data.behavior.duration = 30
    character_data.behavior.behavior_id = constant.Behavior.PLAY_PIANO
    character_data.state = constant.CharacterStatus.STATUS_PLAY_PIANO
    update.game_update_flow(30)


@handle_instruct.add_instruct(
    constant.Instruct.PLAY_GUITAR,
    constant.InstructType.PERFORM,
    _("弹吉他"),
    {constant.Premise.HAVE_GUITAR},
)
def handle_play_guitar():
    """处理弹吉他指令"""
    character.init_character_behavior_start_time(0, cache.game_time)
    character_data = cache.character_data[0]
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.PLAY_GUITAR
    character_data.state = constant.CharacterStatus.STATUS_PLAY_GUITAR
    update.game_update_flow(10)
