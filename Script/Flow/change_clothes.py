from Script.Panel import change_clothes_panel
from Script.Core import flow_handle, cache_contorl, game_config, py_cmd
from Script.Design import clothing


def change_character_clothes():
    """
    更换角色服装流程
    """
    character_id = cache_contorl.character_data[0].target_character_id
    character_clothing_data = cache_contorl.character_data[
        character_id
    ].clothing
    change_clothes_panel.see_character_wear_clothes_info(character_id)
    cmd_list_1 = change_clothes_panel.see_character_wear_clothes(
        character_id, True
    )
    start_id = len(character_clothing_data.keys())
    input_s = change_clothes_panel.see_character_wear_clothes_cmd(start_id)
    input_s = cmd_list_1 + input_s
    yrn = flow_handle.askfor_all(input_s)
    py_cmd.clr_cmd()
    if yrn == str(start_id):
        cache_contorl.now_flow_id = "main"
    else:
        clothing_type = list(clothing.clothing_type_text_list.keys())[int(yrn)]
        see_character_clothes_list(clothing_type)


def see_character_clothes_list(clothing_type: str):
    """
    查看角色服装列表流程
    Keyword arguments:
    clothing_type -- 服装类型
    """
    clothing_type_list = list(clothing.clothing_type_text_list.keys())
    while True:
        now_clothing_type_index = clothing_type_list.index(clothing_type)
        up_type_id = now_clothing_type_index - 1
        if now_clothing_type_index == 0:
            up_type_id = len(clothing_type_list) - 1
        next_type_id = now_clothing_type_index + 1
        if now_clothing_type_index == len(clothing_type_list) - 1:
            next_type_id = 0
        up_type = clothing_type_list[up_type_id]
        next_type = clothing_type_list[next_type_id]
        character_id = cache_contorl.character_data[0].target_character_id
        change_clothes_panel.see_character_clothes_info(character_id)
        page_max = get_character_clothes_page_max(character_id, clothing_type)
        input_s = change_clothes_panel.see_character_clothes_panel(
            character_id, clothing_type, page_max
        )
        start_id = len(input_s)
        input_s += change_clothes_panel.see_character_clothes_cmd(
            start_id, clothing_type
        )
        yrn = flow_handle.askfor_all(input_s)
        yrn = int(yrn)
        py_cmd.clr_cmd()
        now_page_id = int(
            cache_contorl.panel_state["SeeCharacterClothesPanel"]
        )
        if yrn == start_id:
            clothing_type = up_type
        elif yrn == start_id + 1:
            if now_page_id == 0:
                cache_contorl.panel_state["SeeCharacterClothesPanel"] = str(
                    page_max
                )
            else:
                cache_contorl.panel_state["SeeCharacterClothesPanel"] = str(
                    now_page_id - 1
                )
        elif yrn == start_id + 2:
            break
        elif yrn == start_id + 3:
            if now_page_id == page_max:
                cache_contorl.panel_state["SeeCharacterClothesPanel"] = "0"
            else:
                cache_contorl.panel_state["SeeCharacterClothesPanel"] = str(
                    now_page_id + 1
                )
        elif yrn == start_id + 4:
            clothing_type = next_type
        else:
            clothing_id = list(
                cache_contorl.character_data[character_id]
                .clothing[clothing_type]
                .keys()
            )[yrn]
            ask_see_clothing_info(clothing_type, clothing_id, character_id)


def ask_see_clothing_info(
    clothing_type: str, clothing_id: str, character_id: str
):
    """
    确认查看服装详细信息流程
    Keyword arguments:
    clothing_type -- 服装类型
    clothing_id -- 服装id
    character_id -- 角色id
    """
    wear_clothing_judge = False
    if (
        clothing_id
        == cache_contorl.character_data[character_id].put_on[clothing_type]
    ):
        wear_clothing_judge = True
    yrn = int(
        change_clothes_panel.ask_see_clothing_info_panel(wear_clothing_judge)
    )
    if yrn == 0:
        if wear_clothing_judge:
            cache_contorl.character_data[character_id].put_on[
                clothing_type
            ] = ""
        else:
            cache_contorl.character_data[character_id].put_on[
                clothing_type
            ] = clothing_id
    elif yrn == 1:
        see_clothing_info(character_id, clothing_type, clothing_id)


def see_clothing_info(character_id: str, clothing_type: str, clothing_id: str):
    """
    查看服装详细信息的流程
    Keyword arguments:
    character_id -- 角色id
    clothing_type -- 服装类型
    clothing_id -- 服装id
    """
    clothing_list = list(
        cache_contorl.character_data[character_id]
        .clothing[clothing_type]
        .keys()
    )
    while True:
        wear_clothing_judge = False
        if (
            clothing_id
            == cache_contorl.character_data[character_id].put_on[clothing_type]
        ):
            wear_clothing_judge = True
        now_clothing_index = clothing_list.index(clothing_id)
        change_clothes_panel.see_clothing_info_panel(
            character_id, clothing_type, clothing_id, wear_clothing_judge
        )
        yrn = int(
            change_clothes_panel.see_clothing_info_ask_panel(
                wear_clothing_judge
            )
        )
        if yrn == 0:
            if now_clothing_index == 0:
                clothing_id = clothing_list[-1]
            else:
                clothing_id = clothing_list[now_clothing_index - 1]
        elif yrn == 1:
            if wear_clothing_judge:
                cache_contorl.character_data[character_id].put_on[
                    clothing_type
                ] = ""
            else:
                cache_contorl.character_data[character_id].put_on[
                    clothing_type
                ] = clothing_id
        elif yrn == 2:
            break
        elif yrn == 3:
            if clothing_id == clothing_list[-1]:
                clothing_id = clothing_list[0]
            else:
                clothing_id = clothing_list[now_clothing_index + 1]


def get_character_clothes_page_max(
    character_id: str, clothing_type: str
) -> int:
    """
    计算角色某类型服装列表页数
    Keyword arguments:
    character_id -- 角色Id
    clothing_type -- 服装类型
    """
    clothing_max = len(
        cache_contorl.character_data[character_id]
        .clothing[clothing_type]
        .keys()
    )
    page_index = game_config.see_character_clothes_max
    if clothing_max < page_index:
        return 0
    elif not clothing_max % page_index:
        return clothing_max / page_index - 1
    return int(clothing_max / page_index)
