from Script.Core import text_loading, cache_contorl, py_cmd, era_print
from Script.Design import attr_text, cmd_button_queue

def see_character_item_panel(character_id: int) -> list:
    """
    查看角色背包道具列表面板
    Keyword arguments:
    character_id -- 角色Id
    """
    era_print.normal_print(
        attr_text.get_see_attr_panel_head_character_info(character_id)
    )
    era_print.restart_line_print(".")
    if character_id != 0:
        era_print.line_feed_print(
            text_loading.get_text_data(text_loading.MESSAGE_PATH, "37")
        )
        return []
    character_item_data = cache_contorl.character_data["character"][character_id].item
    if len(character_item_data) == 0:
        era_print.line_feed_print(
            text_loading.get_text_data(text_loading.MESSAGE_PATH, "36")
        )
        return []
    now_page_id = int(cache_contorl.panel_state["SeeCharacterItemListPanel"])
    now_page_max = game_config.see_character_item_max
    now_page_start_id = now_page_id * now_page_max
    now_page_end_id = now_page_start_id + now_page_max
    if now_page_end_id > len(character_item_data.keys()):
        now_page_end_id = len(character_item_data.keys())
    input_s = []
    index = 0
    for i in range(now_page_start_id, now_page_end_id):
        item_id = list(character_item_data.keys())[i]
        item_data = character_item_data[item_id]
        item_text = (
            item_data["ItemName"]
            + " "
            + text_loading.get_text_data(text_loading.STAGE_WORD_PATH, "136")
            + str(item_data["ItemNum"])
        )
        if character_id == "0":
            id_info = cmd_button_queue.id_index(index)
            cmd_text = id_info + draw_text
            py_cmd.pcmd(cmd_text, index, None)
        else:
            era_print.normal_print(draw_text)
        index += 1
        input_s.append(str(index))
        era_print.line_feed_print()
    return input_s


def see_character_item_info_panel(character_id: str, item_id: str):
    """
    用于查看角色道具信息的面板
    Keyword arguments:
    character_id -- 角色Id
    item_id -- 道具Id
    """
    title_text = text_loading.get_text_data(text_loading.STAGE_WORD_PATH, "38")
    era_print.little_title_print(title_text)
    era_print.normal_print(
        attr_text.get_see_attr_panel_head_character_info(character_id)
    )
    era_print.restart_line_print(".")
    item_data = cache_contorl.character_data["character"][character_id].item[item_id]
    era_print.line_feed_print(
        text_loading.get_text_data(text_loading.STAGE_WORD_PATH, 128)
        + item_data["ItemName"]
    )
    era_print.line_feed_print(
        text_loading.get_text_data(text_loading.STAGE_WORD_PATH, "131")
        + item_data["ItemInfo"]
    )
