from Script.Core import (
    cache_contorl,
    game_config,
    flow_handle,
    py_cmd,
    text_loading,
)
from Script.Panel import wear_item_panel


def scene_see_character_wear_item(character_id: int):
    """
    在场景中查看角色穿戴道具列表的流程
    Keyword arguments:
    character_id -- 角色Id
    """
    while 1:
        now_input_s = wear_item_panel.see_character_wear_item_panel_for_player(
            character_id
        )
        now_yrn = flow_handle.askfor_all(now_input_s)
        if now_yrn == now_input_s[:-1]:
            cache_contorl.now_flow_id = "main"
            break


def wear_character_item():
    """
    查看并更换角色穿戴道具流程
    """
    character_id = cache_contorl.character_data["character_id"]
    while 1:
        input_s = wear_item_panel.see_character_wear_item_panel_for_player(
            character_id
        )
        start_id = len(input_s)
        input_s += wear_item_panel.see_character_wear_item_cmd_panel(start_id)
        yrn = flow_handle.askfor_all(input_s)
        py_cmd.clr_cmd()
        if yrn == str(len(input_s) - 1):
            cache_contorl.now_flow_id = "main"
            break
        else:
            wear_item_info_text_data = text_loading.get_text_data(
                text_loading.STAGE_WORD_PATH, "49"
            )
            change_wear_item(list(wear_item_info_text_data.keys())[int(yrn)])


def change_wear_item(item_type: str) -> bool:
    """
    更换角色穿戴道具流程
    Keyword arguments:
    item_type -- 道具类型
    """
    character_id = cache_contorl.character_data["character_id"]
    max_page = get_character_wear_item_page_max(character_id)
    input_s = wear_item_panel.see_character_wear_item_list_panel(
        character_id, item_type, max_page
    )
    if input_s == []:
        return
    yrn = flow_handle.askfor_all(input_s)
    if yrn == input_s[:-1]:
        return
    else:
        cache_contorl.character_data["character"][character_id].wear_item[
            "Wear"
        ][item_type] = list(
            cache_contorl.character_data["character"][character_id]
            .wear_item["Item"][item_type]
            .keys()
        )[
            int(yrn)
        ]


def get_character_wear_item_page_max(character_id: str):
    """
    计算角色可穿戴道具列表页数
    Keyword arguments:
    character_id -- 角色Id
    """
    wear_item_max = len(
        cache_contorl.character_data["character"][character_id].wear_item[
            "Item"
        ]
    )
    page_index = game_config.see_character_wearitem_max
    if wear_item_max - page_index < 0:
        return 0
    elif wear_item_max % page_index == 0:
        return wear_item_max / page_index - 1
    else:
        return int(wear_item_max / page_index)
