from Script.Design import character_behavior, game_time


def game_update_flow():
    """
    游戏流程刷新
    """
    game_time.init_school_course_time_status()
    character_behavior.init_character_behavior()
