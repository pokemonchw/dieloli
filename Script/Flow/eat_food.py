from Script.Design import cooking
from Script.Panel import eat_food_panel
from Script.Core import flow_handle, cache_contorl, game_type, game_config


def eat_food() -> (bool, game_type.Food):
    """
    食用食物指令处理流程
    Return arguments:
    bool -- 成功选择要食用的食物
    game_type.Food -- 要食用的食物
    """
    now_panel = "StapleFood"
    while 1:
        eat_food_panel.see_food_bag_head()
        head_buttons = eat_food_panel.see_food_bag_head_cmd(now_panel)
        food_list = cooking.get_character_food_bag_type_list_buy_food_type(0, now_panel)
        max_page = get_food_bag_page_max(len(food_list), 1)
        now_page_id = int(cache_contorl.panel_state["SeeFoodBagListByFoodPanel"])
        food_buttons = eat_food_panel.see_food_bag_list(food_list)
        start_id = len(food_buttons)
        tail_buttons = eat_food_panel.see_food_bag_tail_cmd(start_id, max_page, 0)
        ask_for_list = head_buttons + food_buttons + tail_buttons
        yrn = flow_handle.askfor_all(ask_for_list)
        if yrn in head_buttons:
            now_panel = yrn
            cache_contorl.panel_state["SeeFoodBagListByFoodPanel"] = 0
        elif int(yrn) < start_id:
            now_page_max = game_config.food_shop_item_max
            now_page_start_id = now_page_id * now_page_max
            now_food_name_id = now_page_start_id + int(yrn)
            now_food_name = list(food_list.keys())[now_food_name_id]
            judge, now_food_data = eat_food_by_type(food_list[now_food_name])
            if judge:
                return 1, now_food_data
        elif int(yrn) == start_id:
            if now_page_id == 0:
                cache_contorl.panel_state["SeeFoodBagListByFoodPanel"] = max_page
            else:
                cache_contorl.panel_state["SeeFoodBagListByFoodPanel"] -= 1
        elif int(yrn) == start_id + 1:
            cache_contorl.panel_state["SeeFoodBagListByFoodPanel"] = 0
            break
        else:
            if now_page_id == max_page:
                cache_contorl.panel_state["SeeFoodBagListByFoodPanel"] = 0
            else:
                cache_contorl.panel_state["SeeFoodBagListByFoodPanel"] += 1
    return 0, None


def eat_food_by_type(food_list: set) -> (bool, game_type.Food):
    """
    食用食物指令查看指定种类食物列表流程
    Keyword arguments:
    food_list -- 食物uid集合
    Return arguments:
    bool -- 成功选择要食用的食物
    game_type.Food -- 要食用的食物
    """
    food_bag = cache_contorl.character_data[0].food_bag
    food_data = [food_bag[food_uid] for food_uid in food_list]
    while 1:
        eat_food_panel.see_food_bag_head()
        max_page = get_food_bag_page_max(len(food_data), 0)
        now_page_id = int(cache_contorl.panel_state["SeeFoodBagListByFoodTypePanel"])
        food_buttons = eat_food_panel.see_food_bag_list_by_food_uid(max_page, food_data)
        start_id = len(food_buttons)
        tail_buttons = eat_food_panel.see_food_bag_tail_cmd(start_id, max_page, 1)
        askfor_list = food_buttons + tail_buttons
        yrn = int(flow_handle.askfor_all(askfor_list))
        if yrn < start_id:
            now_page_max = game_config.food_shop_item_max
            now_page_start_id = now_page_id * now_page_max
            now_food_id = now_page_start_id + yrn
            now_food = food_data[now_food_id]
            if eat_food_now(now_food):
                return 1, now_food
        elif yrn == start_id:
            if now_page_id == 0:
                cache_contorl.panel_state["SeeFoodBagListByFoodTypePanel"] = max_page
            else:
                cache_contorl.panel_state["SeeFoodBagListByFoodTypePanel"] -= 1
        elif yrn == start_id + 1:
            cache_contorl.panel_state["SeeFoodBagListByFoodTypePanel"] = 0
            break
        else:
            if now_page_id == max_page:
                cache_contorl.panel_state["SeeFoodBagListByFoodTypePanel"] = 0
            else:
                cache_contorl.panel_state["SeeFoodBagListByFoodTypePanel"] += 1
    return 0, None


def eat_food_now(now_food: game_type.Food) -> bool:
    """
    玩家确认食用食物流程
    Keyword arguments:
    now_food -- 食物对象
    Return arguments:
    bool -- 确认选择食用食物校验
    """
    yrn = int(eat_food_panel.eat_food_now_panel(now_food))
    if yrn == 0:
        return 1
    return 0


def get_food_bag_page_max(food_max: int, type_judge: bool) -> int:
    """
    计算食物背包内某类食物页数
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
