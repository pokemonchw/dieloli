import random
from Script.Core import text_loading, cache_contorl


def get_random_nature():
    """
    初始化角色性格
    """
    nature_list = text_loading.get_game_data(text_loading.NATURE_PATH)
    nature_data = {
        b_dimension: random.uniform(0, 100)
        for a_dimension in nature_list
        for b_dimension in nature_list[a_dimension]["Factor"]
    }
    return nature_data
