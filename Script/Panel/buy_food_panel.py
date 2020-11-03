from Script.Core import (
    cache_contorl,
    era_print,
    text_loading,
    constant,
    game_config,
    py_cmd,
    text_handle,
    game_type,
)
from Script.Design import map_handle, cmd_button_queue, cooking


def see_food_shop_head():
    """
    查看食物商店商品列表顶部面板
    """
    scene_position = cache_contorl.character_data[0].position
    scene_position_str = map_handle.get_map_system_path_str_for_list(scene_position)
    scene_name = cache_contorl.scene_data[scene_position_str]["SceneName"]
    era_print.little_title_print(scene_name)


def see_food_shop_head_cmd(now_panel: str) -> list:
    """
    食物商店顶部控制按钮
    Keyword arguments:
    now_panel -- 当前面板id
    Return arguments:
    list -- 监听的按钮列表
    """
    button_data: dict = text_loading.get_text_data(
        constant.FilePath.CMD_PATH, constant.CmdMenu.BUY_FOOD_HEAD_PANEL
    )
    cmd_list = list(button_data.keys())
    return_list = list(button_data.values())
    input_s = cmd_button_queue.option_str(
        "",
        6,
        "center",
        askfor=False,
        cmd_list_data=return_list,
        null_cmd=now_panel,
        return_data=cmd_list,
    )
    return input_s


def see_food_shop_list(type_list: dict) -> list:
    """
    用于查看餐馆出售食物种类的面板
    Keyword arguments:
    type_list -- 种类列表
    Return arguments:
    list -- 监听的按钮列表
    """
    era_print.restart_line_print("+")
    now_page_id = int(cache_contorl.panel_state["SeeFoodShopListByFoodPanel"])
    now_page_max = game_config.food_shop_type_max
    now_page_start_id = now_page_id * now_page_max
    now_page_end_id = now_page_start_id + now_page_max
    if not len(type_list):
        era_print.normal_print(text_loading.get_text_data(constant.FilePath.MESSAGE_PATH, "45"))
        era_print.line_feed_print()
        return []
    if now_page_end_id > len(type_list):
        now_page_end_id = len(type_list)
    text_list = []
    for i in range(now_page_start_id, now_page_end_id):
        food_name = type_list[list(type_list.keys())[i]]
        text_list.append(food_name)
    return cmd_button_queue.option_int("", 4, "left", 1, 0, "center", 0, text_list)


def see_food_shop_list_by_food_type(max_page: int, food_list: list) -> list:
    """
    用于查看餐馆出售食物列表的面板
    Keyword arguments:
    max_page -- 最大页数
    Return arguments:
    list -- 监听的按钮列表
    """
    era_print.restart_line_print("+")
    tag_text_index = 0
    now_page_id = int(cache_contorl.panel_state["SeeFoodShopListByFoodTypePanel"])
    now_page_max = game_config.food_shop_item_max
    now_page_start_id = now_page_id * now_page_max
    now_page_end_id = now_page_start_id + now_page_max
    if not len(food_list):
        era_print.normal_print(text_loading.get_text_data(constant.FilePath.MESSAGE_PATH, "34"))
        era_print.line_feed_print()
        return []
    if now_page_end_id > len(food_list):
        now_page_end_id = len(food_list)
    input_s = []
    text_list = []
    fix_text = ""
    for i in range(now_page_start_id, now_page_end_id):
        now_food = food_list[i]
        if now_food.recipe == -1:
            food_config = text_loading.get_text_data(constant.FilePath.FOOD_PATH, now_food.id)
            food_name = food_config["Name"]
        else:
            food_name = cache_contorl.recipe_data[now_food.recipe].name
        now_index = cmd_button_queue.id_index(tag_text_index)
        food_text = now_index + " " + food_name
        food_text += " " + text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "154")
        if "Hunger" in now_food.feel:
            food_text += str(round(now_food.feel["Hunger"], 2))
        else:
            food_text += "0"
        food_text += " " + text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "153")
        if "Thirsty" in now_food.feel:
            food_text += str(round(now_food.feel["Thirsty"], 2))
        else:
            food_text += "0"
        food_text += (
            " "
            + text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "155")
            + str(now_food.weight)
        )
        food_text += text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "156")
        food_text += " " + text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "152")
        food_text += text_loading.get_text_data(constant.FilePath.STAGE_WORD_PATH, "151")[now_food.quality]
        text_list.append(food_text)
        now_fix_text = text_handle.align(food_text, "center", True)
        if fix_text == "":
            fix_text = now_fix_text
        if len(now_fix_text) < len(fix_text):
            fix_text = now_fix_text
        tag_text_index += 1
    for i in range(tag_text_index):
        now_text = fix_text + text_list[i]
        now_text = text_handle.align(now_text)
        py_cmd.pcmd(now_text, i)
        era_print.normal_print("\n")
        input_s.append(str(i))
        if i < tag_text_index - 1:
            era_print.restart_line_print("*")
    return input_s


def see_food_shop_tail_cmd(start_id: int, max_page: int, type_judge: bool) -> list:
    """
    食物商店底部控制面板
    Keyword arguments:
    start_id -- 按钮的id的开始位置
    max_page -- 最大页数
    type_judge -- 是否是食物类型列表
    Return arguments:
    list -- 监听的按钮列表
    """
    now_page_id = int(cache_contorl.panel_state["SeeFoodShopListByFoodPanel"])
    if type_judge:
        now_page_id = int(cache_contorl.panel_state["SeeFoodShopListByFoodTypePanel"])
    page_text = f"({now_page_id}/{max_page})"
    era_print.page_line_print("-", page_text)
    era_print.normal_print("\n")
    cmd_list = text_loading.get_text_data(constant.FilePath.CMD_PATH, "changeSavePage")
    yrn = cmd_button_queue.option_int(
        None,
        3,
        askfor=False,
        cmd_size="center",
        start_id=start_id,
        cmd_list_data=cmd_list,
    )
    return yrn


def buy_food_now_panel(now_food: game_type.Food) -> int:
    """
    玩家确认购买食物面板
    Keyword arguments:
    now_food -- 食物对象
    Return arguments:
    list -- 监听的按钮列表
    """
    now_text = ""
    if now_food.recipe == -1:
        food_config = text_loading.get_text_data(constant.FilePath.FOOD_PATH, now_food.id)
        now_text = text_loading.get_text_data(constant.FilePath.MESSAGE_PATH, "43").format(
            FoodName=food_config["Name"]
        )
    else:
        food_recipe = cache_contorl.recipe_data[now_food.recipe]
        now_text = text_loading.get_text_data(constant.FilePath.MESSAGE_PATH, "43").format(
            FoodName=food_recipe.name
        )
    era_print.line_feed_print(now_text)
    return cmd_button_queue.option_int(constant.CmdMenu.BUY_FOOD_NOW_PANEL)
