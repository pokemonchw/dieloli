from Script.Panel import buy_food_panel
from Script.Design import cooking
from Script.Core import game_config, flow_handle


def buy_food():
    """
    购买食物指令处理流程
    """
    buy_food_panel.see_food_shop_head()
    now_panel = "StapleFood"
    while 1:
        head_buttons = buy_food_panel.see_food_shop_head_cmd(now_panel)
        max_page = get_food_shop_page_max(now_panel)
        food_buttons = buy_food_panel.see_food_shop_list_by_food_type(
            now_panel, max_page
        )
        tail_buttons = buy_food_panel.see_food_shop_tail_cmd(
            len(food_buttons), max_page
        )
        ask_for_list = head_buttons + food_buttons + tail_buttons
        yrn = flow_handle.askfor_all(ask_for_list)


def get_food_shop_page_max(food_type: str) -> int:
    """
    计算食物商店内某类食物页数
    Keyword arguments:
    food_type -- 食物类型
    Return arguments:
    int -- 页数
    """
    food_max = len(cooking.get_restaurant_food_list_buy_food_type(food_type))
    page_index = game_config.food_shop_item_max
    if food_max < page_index:
        return 0
    elif not food_max % page_index:
        return food_max / page_index - 1
    return int(food_max / page_index)
