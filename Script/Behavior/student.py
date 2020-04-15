from Script.Core import cache_contorl
from Script.Design import game_time
from Script.Behavior import default


# 职业行为总控制
def behavior_init(character_id: int):
    """
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


# 休闲状态行为
def arder_behavior(character_id: int):
    """
    Keyword arguments:
    character_id -- 角色id
    """
    now_week_day = game_time.get_week_date()
    if now_week_day in range(1, 5):
        now_time_slice = game_time.get_now_time_slice()


behavior_list = {"arder": arder_behavior}
