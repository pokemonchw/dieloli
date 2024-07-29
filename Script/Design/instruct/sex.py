from types import FunctionType
from Script.Core import cache_control, game_type, get_text
from Script.Design import update, constant, handle_instruct
from Script.Config import normal_config


cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
_: FunctionType = get_text._
""" 翻译api """
width: int = normal_config.config_normal.text_width
""" 屏幕宽度 """


@handle_instruct.add_instruct(
    constant.Instruct.MASTURBATION,
    constant.InstructType.SEX,
    _("手淫"),
    {
        constant.Premise.NO_WEAR_UNDERPANTS,
        constant.Premise.NO_WEAR_PANTS,
        constant.Premise.IS_NOT_ASEXUAL,
    },
)
def handle_masturbation():
    """处理手淫指令"""
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    character_data.behavior.duration = 10
    character_data.behavior.behavior_id = constant.Behavior.MASTURBATION
    character_data.state = constant.CharacterStatus.STATUS_MASTURBATION
    update.game_update_flow(10)


@handle_instruct.add_instruct(
    constant.Instruct.INVITE_SEX,
    constant.InstructType.SEX,
    _("邀请做爱"),
    {
        constant.Premise.HAVE_TARGET,
    }
)
def handle_missionary_position():
    """处理邀请做爱指令"""
    character_data: game_type.Character = cache.character_data[0]
    character_data.behavior.start_time = cache.game_time
    constant.settle_behavior_effect_data[constant.BehaviorEffect.INTERRUPT_TARGET_ACTIVITY](0,1,None,cache.game_time)
    target_data: game_type.Character = cache.character_data[character_data.target_character_id]
    target_data.behavior.start_time = character_data.behavior.start_time
    character_data.behavior.duration = 1
    character_data.behavior.behavior_id = constant.Behavior.INVITE_SEX
    character_data.state = constant.CharacterStatus.STATUS_INVITE_SEX
    update.game_update_flow(1)
