from Script.Design import game_time
from Script.Core import constant, cache_contorl


def arder_behavior(character_id: int):
    """
    休闲状态行为
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache_contorl.character_data[character_id]
    now_time_slice = game_time.get_now_time_slice(character_id)


behavior_list = {constant.CharacterStatus.STATUS_ARDER: arder_behavior}
