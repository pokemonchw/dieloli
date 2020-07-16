from Script.Panel import buy_food_panel
from Script.Design import cooking
from Script.Core import game_config, flow_handle,cache_contorl,game_type


def buy_food():
    """
    购买食物指令处理流程
    """
    now_panel = "StapleFood"
    while 1:
        buy_food_panel.see_food_shop_head()
        head_buttons = buy_food_panel.see_food_shop_head_cmd(now_panel)
        food_list = cooking.get_restaurant_food_type_list_buy_food_type(now_panel)
        max_page = get_food_shop_page_max(len(food_list),1)
        now_page_id = int(
            cache_contorl.panel_state["SeeFoodShopListByFoodPanel"]
        )
        food_buttons = buy_food_panel.see_food_shop_list(
            food_list
        )
        start_id = len(food_buttons)
        tail_buttons = buy_food_panel.see_food_shop_tail_cmd(
            start_id, max_page,0
        )
        ask_for_list = head_buttons + food_buttons + tail_buttons
        yrn = flow_handle.askfor_all(ask_for_list)
        if yrn in head_buttons:
            now_panel = yrn
            cache_contorl.panel_state["SeeFoodShopListByFoodPanel"] = 0
        elif int(yrn) < start_id:
            now_page_max = game_config.food_shop_item_max
            now_page_start_id = now_page_id * now_page_max
            now_type_id = now_page_start_id + int(yrn)
            now_type = list(food_list.keys())[now_type_id]
            buy_food_by_type(now_type)
        elif int(yrn) == start_id:
            if now_page_id == 0:
                cache_contorl.panel_state["SeeFoodShopListByFoodPanel"] = max_page
            else:
                cache_contorl.panel_state["SeeFoodShopListByFoodPanel"] -= 1
        elif int(yrn) == start_id + 1:
            cache_contorl.panel_state["SeeFoodShopListByFoodPanel"] = 0
            break
        else:
            if now_page_id == max_page:
                cache_contorl.panel_state["SeeFoodShopListByFoodPanel"] = 0
            else:
                cache_contorl.panel_state["SeeFoodShopListByFoodPanel"] += 1


def buy_food_by_type(food_id:str):
    """
    购买食物指令查看指定种类食物列表流程
    Keyword arguments:
    food_id -- 食物种类
    """
    while 1:
        buy_food_panel.see_food_shop_head()
        food_list = list(cache_contorl.restaurant_data[food_id].values())
        max_page = get_food_shop_page_max(len(food_list),0)
        now_page_id = int(cache_contorl.panel_state["SeeFoodShopListByFoodTypePanel"])
        food_buttons = buy_food_panel.see_food_shop_list_by_food_type(max_page,food_list)
        start_id = len(food_buttons)
        tail_buttons = buy_food_panel.see_food_shop_tail_cmd(
            start_id, max_page,1
        )
        askfor_list = food_buttons + tail_buttons
        yrn = int(flow_handle.askfor_all(askfor_list))
        if yrn < start_id:
            now_page_max = game_config.food_shop_item_max
            now_page_start_id = now_page_id * now_page_max
            now_food_id = now_page_start_id + yrn
            now_food = food_list[now_food_id]
            buy_food_now(now_food,food_id)
        elif yrn == start_id:
            if now_page_id == 0:
                cache_contorl.panel_state["SeeFoodShopListByFoodTypePanel"] = max_page
            else:
                cache_contorl.panel_state["SeeFoodShopListByFoodTypePanel"] -= 1
        elif yrn == start_id + 1:
            cache_contorl.panel_state["SeeFoodShopListByFoodTypePanel"] = 0
            break
        else:
            if now_page_id == max_page:
                cache_contorl.panel_state["SeeFoodShopListByFoodTypePanel"] = 0
            else:
                cache_contorl.panel_state["SeeFoodShopListByFoodTypePanel"] += 1


def buy_food_now(now_food:game_type.Food,food_id:str):
    """
    玩家确认购买食物流程
    Keyword arguments:
    now_food -- 食物对象
    food_id -- 食物对象在餐馆中的配置id
    """
    yrn = int(buy_food_panel.buy_food_now_panel(now_food))
    if yrn == 0:
        cache_contorl.character_data[0].food_bag[now_food.uid] = now_food
        del cache_contorl.restaurant_data[food_id][now_food.uid]
    elif yrn == 1:
        if now_food.weight > 100:
            new_food = cooking.separate_weight_food(now_food,100)
            cache_contorl.character_data[0].food_bag[new_food.uid] = new_food
        else:
            cache_contorl.character_data[0].food_bag[now_food.uid] = now_food
            del cache_contorl.restaurant_data[food_id][now_food.uid]


def get_food_shop_page_max(food_max:int,type_judge:bool) -> int:
    """
    计算食物商店内某类食物页数
    Keyword arguments:
    food_max -- 食物数量
    type_judge -- 食物类型校验
    Return arguments:
    int -- 页数
    """
    page_index = game_config.food_shop_item_max
    if type_judge:
        page_index = game_config.food_shop_type_max
    if food_max < page_index:
        return 0
    elif not food_max % page_index:
        return food_max / page_index - 1
    return int(food_max / page_index)
