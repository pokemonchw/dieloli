from Script.Design import handle_achieve, constant
from Script.Core import cache_control


@handle_achieve.add_achieve(constant.Achieve.START_GAME)
def handle_start_game():
    """记录开始游戏成就"""
    return 1


@handle_achieve.add_achieve(constant.Achieve.END_GAME)
def handle_end_game():
    """初次死亡"""
    if cache_control.achieve.first_dead:
        return 1
    return 0


@handle_achieve.add_achieve(constant.Achieve.FIRST_BLOOD)
def handle_first_blood():
    """第一次做爱"""
    if cache_control.achieve.first_blood:
        return 1
    return 0


@handle_achieve.add_achieve(constant.Achieve.DROWNED_GIRL)
def handle_drowned_girl():
    """喝下了娘溺泉的水"""
    if cache_control.achieve.drowned_girl:
        return 1
    return 0


@handle_achieve.add_achieve(constant.Achieve.ONE_HUNDRED_AND_EIGHT_THOUSAND_SORROWS)
def handle_one_hundred_and_eight_thousand_sorrows():
    """见证了108000人的人生"""
    if cache_control.achieve.create_npc_index >= 108000:
        return 1
    return 0


@handle_achieve.add_achieve(constant.Achieve.FIRST_WEAR_CLOTHES)
def handle_first_wear_clothes():
    """记录初次穿上衣服"""
    if cache_control.achieve.first_wear_clothes:
        return 1
    return 0
