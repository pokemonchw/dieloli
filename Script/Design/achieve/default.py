from Script.Design import handle_achieve, constant
from Script.Core import cache_control


@handle_achieve.add_achieve(constant.Achieve.START_GAME)
def handle_start_game():
    """记录开始游戏成就"""
    return 1

