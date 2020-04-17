from Script.Core import cache_contorl, constant
from Script.Design import game_time
from Script.Behavior import default
from Script.Behavior.Action import default_action


def behavior_init(character_id: int):
    """
    职业行为总控制
    Keyword arguments:
    character_id -- 角色id
    """
    character_state = cache_contorl.character_data["character"][
        character_id
    ].state
    if character_state in behavior_list:
        behavior_list[character_state](character_id)
    else:
        default.behavior_list[character_state](character_id)


def arder_behavior(character_id: int):
    """
    休闲状态行为
    Keyword arguments:
    character_id -- 角色id
    """
    character_data = cache_contorl.character_data["character"][character_id]
    now_time_slice = game_time.get_now_time_slice(character_id)
    if now_time_slice["InCourse"]:
        if character_data.position != character_data.classroom:
            default_action.move_action(character_id, character_data.classroom)


behavior_list = {constant.CharacterStatus.STATUS_ARDER: arder_behavior}
