from Script.Core import game_config, game_init, cache_contorl
from Script.Panel import see_character_list_panel

character_page_show = int(game_config.character_list_show)


def see_character_list_func():
    """
    用于查看角色列表的流程
    """
    while True:
        max_page = get_character_list_page_max()
        input_s = []
        see_character_list_panel_input = (
            see_character_list_panel.see_character_list_panel(max_page)
        )
        start_id = len(see_character_list_panel_input)
        input_s = input_s + see_character_list_panel_input
        ask_for_see_character_list_panel_input = (
            see_character_list_panel.ask_for_see_character_list_panel(start_id)
        )
        input_s = input_s + ask_for_see_character_list_panel_input
        yrn = game_init.askfor_all(input_s)
        yrn = str(yrn)
        character_id_list = cache_contorl.character_data.keys()
        page_id = int(cache_contorl.panel_state["SeeCharacterListPanel"])
        if yrn == str(start_id):
            if page_id == 0:
                cache_contorl.panel_state["SeeCharacterListPanel"] = str(max_page)
            else:
                page_id = str(page_id - 1)
                cache_contorl.panel_state["SeeCharacterListPanel"] = page_id
        elif yrn == str(start_id + 1):
            cache_contorl.character_data[0].target_character_id = 0
            cache_contorl.panel_state["SeeCharacterListPanel"] = "0"
            cache_contorl.now_flow_id = cache_contorl.old_flow_id
            break
        elif yrn == str(start_id + 2):
            if page_id == max_page:
                cache_contorl.panel_state["SeeCharacterListPanel"] = "0"
            else:
                page_id = str(page_id + 1)
                cache_contorl.panel_state["SeeCharacterListPanel"] = page_id
        elif int(yrn) + character_page_show * page_id in character_id_list:
            yrn = int(yrn) + character_page_show * page_id
            cache_contorl.character_data[0].target_character_id = yrn
            cache_contorl.now_flow_id = "see_character_attr"
            cache_contorl.too_old_flow_id = cache_contorl.old_flow_id
            cache_contorl.old_flow_id = "see_character_list"
            break


def get_character_list_page_max():
    """
    计算角色列表总页数，公式为角色总数/每页显示角色数
    """
    character_max = len(cache_contorl.character_data) - 1
    if character_max - character_page_show < 0:
        return 0
    elif character_max % character_page_show == 0:
        return character_max / character_page_show - 1
    else:
        return int(character_max / character_page_show)
