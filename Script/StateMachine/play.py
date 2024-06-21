from Script.Design import handle_state_machine, constant
from Script.Core import cache_control, game_type
from Script.Config import game_config

cache: game_type.Cache = cache_control.cache


@handle_state_machine.add_state_machine(constant.StateMachine.RUN)
def character_run(character_id: int):
    """
    角色在场景里跑步
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.behavior.behavior_id = constant.Behavior.RUN
    character_data.behavior.duration = 2
    character_data.state = constant.CharacterStatus.STATUS_RUN

