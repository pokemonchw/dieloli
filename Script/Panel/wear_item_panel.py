from Script.Core import text_loading, cache_contorl, era_print, game_config
from Script.Design import attr_text, cmd_button_queue


def see_character_wear_item_panel_for_player(character_id: int) -> list:
    """
    用于场景中查看穿戴道具列表的控制面板
    Keyword arguments:
    character_id -- 角色Id
    change_button -- 将角色穿戴道具列表绘制成按钮的开关
    """
    era_print.little_title_print(
        text_loading.get_text_data(text_loading.STAGE_WORD_PATH, "40")
    )
    era_print.normal_print(
        attr_text.get_see_attr_panel_head_character_info(character_id)
    )
    era_print.restart_line_print(".")
    if character_id == 0:
        return see_character_wear_item_panel(character_id, True)
    else:
        return see_character_wear_item_panel(character_id, False)


def see_character_wear_item_panel(character_id: int, change_button: bool) -> list:
    """
    用于查看角色穿戴道具列表的面板
    Keyword arguments:
    character_id -- 角色Id
    change_button -- 将角色穿戴道具列表绘制成按钮的开关
    """
    wear_item_info_text_data = text_loading.get_text_data(
        text_loading.STAGE_WORD_PATH, "49"
    )
    wear_data = cache_contorl.character_data["character"][character_id].wear_item[
        "Wear"
    ]
    wear_item_text_data = {}
    item_data = cache_contorl.character_data["character"][character_id].wear_item[
        "Item"
    ]
    wear_item_button_list = []
    input_s = []
    for wear_type in wear_data:
        now_wear_data = wear_data[wear_type]
        if now_wear_data == {}:
            wear_item_button_list.append(
                wear_item_info_text_data[wear_type]
                + ":"
                + text_loading.get_text_data(text_loading.STAGE_WORD_PATH, "117")
            )
        else:
            wear_text = ""
            for wear_id in now_wear_data:
                wear_text += "[" + item_data[wear_type][wear_id]["Name"] + "]"
            wear_item_button_list.append(
                wear_item_info_text_data[wear_type] + ":" + wear_text
            )
            wear_item_text_data[wear_type] = item_data[wear_type][wear_id]["Name"]
    if change_button:
        input_s = [str(i) for i in range(len(wear_data))]
        cmd_button_queue.option_int(
            None, 4, "left", True, False, "center", 0, wear_item_button_list,
        )
    else:
        era_print.list_print(wear_item_button_list, 4, "center")
    return input_s


def see_character_wear_item_list_panel(
    character_id: int, item_type: str, max_page: int
) -> list:
    """
    用于查看角色可穿戴道具列表的面板
    Keyword arguments:
    character_id -- 用户Id
    item_type -- 道具类型
    max_page -- 道具列表最大页数
    """
    era_print.line_feed_print()
    character_wear_item_data = [
        item
        for item in cache_contorl.character_data["character"][character_id].wear_item[
            "Item"
        ]
        if item in cache_contorl.wear_item_type_data[item_type]
    ]
    now_page_id = int(cache_contorl.panel_state["SeeCharacterWearItemListPanel"])
    now_page_max = game_config.see_character_wearitem_max
    now_page_start_id = now_page_id * now_page_max
    now_page_end_id = now_page_start_id + now_page_max
    if character_wear_item_data == []:
        era_print.line_feed_print(
            text_loading.get_text_data(text_loading.MESSAGE_PATH, "38")
        )
        return []
    if now_page_end_id > len(character_wear_item_data.keys()):
        now_page_end_id = len(character_wear_item_data.keys())


def see_character_wear_item_cmd_panel(start_id: int) -> list:
    """
    查看角色已穿戴道具列表的控制面板
    Keyword arguments:
    start_id -- 命令起始Id
    """
    era_print.restart_line_print()
    yrn = cmd_button_queue.option_int(
        cmd_button_queue.SEE_CHARACTER_WEAR_CHOTHES,
        cmd_size="center",
        askfor=False,
        start_id=start_id,
    )
    return yrn
