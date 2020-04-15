from Script.Design import game_time


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
