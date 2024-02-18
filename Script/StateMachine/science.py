from Script.Design import handle_state_machine, constant
from Script.Core import cache_control, game_type

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


@handle_state_machine.add_state_machine(constant.StateMachine.SEE_STAR)
def character_see_star(character_id: int):
    """
    看星星
    Keyword arguments:
    character_id -- 角色id
    """
    character_data: game_type.Character = cache.character_data[character_id]
    character_data.behavior.behavior_id = constant.Behavior.SEE_STAR
    character_data.behavior.duration = 10
    character_data.state = constant.CharacterStatus.STATUS_SEE_STAR

