from Script.Core import cache_contorl, constant
from Script.Design import game_time, character_behavior


@character_behavior.add_behavior(
    "Teacher", constant.CharacterStatus.STATUS_ARDER
)
def arder_behavior(character_id: int):
    """
    教师休闲状态行为
    Keyword arguments:
    character_id -- 角色id
    """
    now_time_slice = game_time.get_now_time_slice(character_id)
