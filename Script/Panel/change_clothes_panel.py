from Script.Core import (
    era_print,
    text_loading,
    cache_contorl,
    py_cmd,
    text_handle,
    game_config,
)
from Script.Design import attr_text, cmd_button_queue, clothing


def see_character_wear_clothes_info(character_id: str):
    """
    查看角色已穿戴服装列表顶部面板
    Keyword arguments:
    character_id -- 角色id
    """
    scene_info = text_loading.get_text_data(text_loading.STAGE_WORD_PATH, "79")
    era_print.little_title_print(scene_info)
    character_info = attr_text.get_character_abbreviations_info(character_id)
    era_print.normal_print(character_info)


def see_character_wear_clothes(character_id: str, change_button: bool):
    """
    查看角色穿戴服装列表面板
    Keyword arguments:
    character_id -- 角色id
    change_button -- 将服装列表绘制成按钮的开关
    """
    character_clothing_data = cache_contorl.character_data["character"][
        character_id
    ].clothing
    character_put_on_list = cache_contorl.character_data["character"][
        character_id
    ].put_on
    clothing_text_data = {}
    tag_text_index = 0
    for i in range(len(clothing.clothing_type_text_list.keys())):
        clothing_type = list(clothing.clothing_type_text_list.keys())[i]
        clothing_id = character_put_on_list[clothing_type]
        if clothing_id == "":
            clothing_text_data[clothing_type] = "None"
        else:
            clothing_data = character_clothing_data[clothing_type][clothing_id]
            clothing_text = (
                clothing.clothing_type_text_list[clothing_type]
                + ":"
                + clothing_data["Evaluation"]
                + clothing_data["Tag"]
                + clothing_data["Name"]
            )
            clothing_text_data[clothing_text] = {}
            for tag in clothing.clothing_tag_text_list:
                tag_text = clothing.clothing_tag_text_list[tag] + str(
                    clothing_data[tag]
                )
                clothing_text_data[clothing_text][tag_text] = 0
                now_tag_text_index = text_handle.get_text_index(tag_text)
                if now_tag_text_index > tag_text_index:
                    tag_text_index = now_tag_text_index
    long_clothing_text_index = text_handle.get_text_index(
        max(clothing_text_data.keys(), key=text_handle.get_text_index)
    )
    i = 0
    input_s = []
    for clothing_text in clothing_text_data:
        draw_text = ""
        era_print.little_line_print()
        if clothing_text_data[clothing_text] == "None":
            draw_text = (
                clothing.clothing_type_text_list[clothing_text]
                + ":"
                + text_loading.get_text_data(text_loading.STAGE_WORD_PATH, "117")
            )
        else:
            now_clothing_text_index = text_handle.get_text_index(clothing_text)
            draw_text += clothing_text + " "
            if now_clothing_text_index < long_clothing_text_index:
                draw_text += " " * (long_clothing_text_index - now_clothing_text_index)
            for tag_text in clothing_text_data[clothing_text]:
                now_tag_text_index = text_handle.get_text_index(tag_text)
                if now_tag_text_index < tag_text_index:
                    draw_text += (
                        " " + tag_text + " " * (tag_text_index - now_tag_text_index)
                    )
                else:
                    draw_text += " " + tag_text
        if change_button:
            id_info = cmd_button_queue.id_index(i)
            cmd_text = id_info + draw_text
            py_cmd.pcmd(cmd_text, i, None)
        else:
            era_print.normal_print(draw_text)
        input_s.append(f"{i}")
        i += 1
        era_print.normal_print('\n')
    return input_s


def see_character_wear_clothes_cmd(start_id: int) -> str:
    """
    用于控制查看角色已装备服装列表面板的命令菜单
    """
    era_print.list_print()()
    yrn = cmd_button_queue.optionint(
        cmd_button_queue.SEE_CHARACYTER_CLOTHES,
        id_size="center",
        askfor=False,
        start_id=start_id,
    )
    return yrn


def see_character_clothes_panel(character_id: str, clothing_type: str, max_page: int):
    """
    用于查看角色服装列表的面板
    Keyword arguments:
    character_id -- 角色id
    clothing_type -- 服装类型
    max_page -- 服装列表最大页数
    """
    era_print.line_feed_print()
    character_clothing_data = cache_contorl.character_data["character"][
        character_id
    ].clothing[clothing_type]
    character_put_on_list = cache_contorl.character_data["character"][
        character_id
    ].put_on
    clothing_text_data = {}
    tag_text_index = 0
    now_page_id = int(cache_contorl.panel_state["SeeCharacterClothesPanel"])
    now_page_max = game_config.see_character_clothes_max
    now_page_start_id = now_page_id * now_page_max
    now_page_end_id = now_page_start_id + now_page_max
    if character_clothing_data == {}:
        era_print.normal_print(
            text_loading.get_text_data(text_loading.MESSAGE_PATH, "34")
        )
        era_print.line_feed_print()
        return []
    if now_page_end_id > len(character_clothing_data.keys()):
        now_page_end_id = len(character_clothing_data.keys())
    pass_id = None
    for i in range(now_page_start_id, now_page_end_id):
        clothing_id = list(character_clothing_data.keys())[i]
        if (
            clothing_id
            == cache_contorl.character_data["character"][character_id].put_on[
                clothing_type
            ]
        ):
            pass_id = i - now_page_start_id
        clothing_data = character_clothing_data[clothing_id]
        clothing_text = (
            clothing_data["Evaluation"] + clothing_data["Tag"] + clothing_data["Name"]
        )
        clothing_text_data[clothing_text] = {}
        for tag in clothing.clothing_tag_text_list:
            tag_text = clothing.clothing_tag_text_list[tag] + str(clothing_data[tag])
            clothing_text_data[clothing_text][tag_text] = 0
            now_tag_text_index = text_handle.get_text_index(tag_text)
            if now_tag_text_index == now_tag_text_index:
                tag_text_index = now_tag_text_index
    long_clothing_text_index = text_handle.get_text_index(
        max(clothing_text_data.keys(), key=text_handle.get_text_index)
    )
    i = 0
    input_s = []
    for clothing_text in clothing_text_data:
        draw_text = ""
        era_print.little_line_print()
        now_clothing_text_index = text_handle.get_text_index(clothing_text)
        draw_text += clothing_text + " "
        if now_clothing_text_index < long_clothing_text_index:
            draw_text += " " * (long_clothing_text_index - now_clothing_text_index)
        for tag_text in clothing_text_data[clothing_text]:
            now_tag_text_index = text_handle.get_text_index(tag_text)
            if now_tag_text_index < tag_text_index:
                draw_text += (
                    " " + tag_text + " " * (tag_text_index - now_tag_text_index)
                )
            else:
                draw_text += " " + tag_text
        if i == pass_id:
            draw_text += " " + text_loading.get_text_data(
                text_loading.STAGE_WORD_PATH, "125"
            )
        id_info = cmd_button_queue.id_index(i)
        cmd_text = id_info + draw_text
        input_s.append(f"{i}")
        py_cmd.pcmd(cmd_text, i, None)
        era_print.line_feed_print()
        i += 1
    era_print.line_feed_print()
    page_text = "(" + str(now_page_id) + "/" + str(max_page) + ")"
    era_print.page_line_print(sample="-", string=page_text)
    era_print.line_feed_print()
    return input_s


def see_character_clothes_info(character_id: str):
    """
    查看角色服装列表顶部面板
    Keyword arguments:
    character_id -- 角色id
    """
    scene_info = text_loading.get_text_data(text_loading.STAGE_WORD_PATH, "101")
    era_print.little_title_print(scene_info)
    character_info = attr_text.get_character_abbreviations_info(character_id)
    era_print.normal_print(character_info)


def see_character_wear_clothes_cmd(start_id: int) -> str:
    """
    用于控制查看角色服装列表面板的命令菜单
    """
    era_print.line_feed_print()
    yrn = cmd_button_queue.option_int(
        cmd_button_queue.SEE_CHARACTER_WEAR_CHOTHES,
        cmd_size="center",
        askfor=False,
        start_id=start_id,
    )
    return yrn


def see_character_clothes_cmd(start_id: int, now_clothing_type: str) -> str:
    """
    用于控制查看角色服装列表面板的命令菜单
    Keyword arguments:
    start_id -- cmd命令的初始Id
    now_clothing_type -- 当前列表的服装类型
    """
    era_print.line_feed_print()
    clothing_type_list = list(clothing.clothing_type_text_list.keys())
    cmd_list = text_loading.get_text_data(
        text_loading.CMD_PATH, cmd_button_queue.SEE_CHARACYTER_CLOTHES
    )
    now_clothing_type_index = clothing_type_list.index(now_clothing_type)
    up_type_id = now_clothing_type_index - 1
    if now_clothing_type_index == 0:
        up_type_id = len(clothing_type_list) - 1
    next_type_id = now_clothing_type_index + 1
    if now_clothing_type_index == len(clothing_type_list) - 1:
        next_type_id = 0
    up_type_text = [clothing.clothing_type_text_list[clothing_type_list[up_type_id]]]
    next_type_text = [
        clothing.clothing_type_text_list[clothing_type_list[next_type_id]]
    ]
    cmd_list = up_type_text + cmd_list + next_type_text
    yrn = cmd_button_queue.option_int(
        None,
        5,
        cmd_size="center",
        askfor=False,
        start_id=start_id,
        cmd_list_data=cmd_list,
    )
    return yrn


def ask_see_clothing_info_panel(wear_clothing_judge: bool) -> str:
    """
    用于询问查看或穿戴服装的面板
    Keyword arguments:
    wear_clothing_judge -- 当前服装穿戴状态
    """
    era_print.line_feed_print()
    titile_message = text_loading.get_text_data(text_loading.MESSAGE_PATH, "35")
    cmd_data = text_loading.get_text_data(
        text_loading.CMD_PATH, cmd_button_queue.ASK_SEE_CLOTHING_INFO_PANEL
    ).copy()
    if wear_clothing_judge:
        del cmd_data["0"]
    else:
        del cmd_data["1"]
    cmd_list = list(cmd_data.values())
    return cmd_button_queue.option_int(None, cmd_list_data=cmd_list)


def see_clothing_info_panel(
    character_id: str, clothing_type: str, clothing_id: str, wear_clothing_judge: bool
):
    """
    查看服装详细信息面板
    Keyword arguments:
    character_id -- 角色id
    clothing_type -- 服装类型
    clothing_id -- 服装id
    """
    era_print.little_title_print(
        text_loading.get_text_data(text_loading.STAGE_WORD_PATH, "126")
    )
    clothing_data = cache_contorl.character_data["character"][character_id].clothing[
        clothing_type
    ][clothing_id]
    info_list = []
    clothing_name = clothing_data["Name"]
    if wear_clothing_judge:
        clothing_name += " " + text_loading.get_text_data(
            text_loading.STAGE_WORD_PATH, "125"
        )
    info_list.append(
        text_loading.get_text_data(text_loading.STAGE_WORD_PATH, "128") + clothing_name
    )
    clothing_type_text = clothing.clothing_type_text_list[clothing_type]
    info_list.append(
        text_loading.get_text_data(text_loading.STAGE_WORD_PATH, "129")
        + clothing_type_text
    )
    evaluation_text = (
        text_loading.get_text_data(text_loading.STAGE_WORD_PATH, "131")
        + clothing_data["Evaluation"]
    )
    info_list.append(evaluation_text)
    era_print.list_print(info_list, 3, "center")
    era_print.son_title_print(
        text_loading.get_text_data(text_loading.STAGE_WORD_PATH, "130")
    )
    tag_text_list = []
    for tag in clothing.clothing_tag_text_list:
        tag_text = clothing.clothing_tag_text_list[tag]
        tag_text += str(clothing_data[tag])
        tag_text_list.append(tag_text)
    era_print.list_print(tag_text_list, 4, "center")
    era_print.son_title_print(
        text_loading.get_text_data(text_loading.STAGE_WORD_PATH, "127")
    )
    era_print.normal_print(clothing_data["Describe"])


def see_clothing_info_ask_panel(wear_clothing_judge: bool) -> str:
    """
    查看服装详细信息的控制面板
    Keyword arguments:
    wear_clothing_judge -- 服装穿戴状态
    """
    era_print.line_feed_print()
    cmd_data = text_loading.get_text_data(
        text_loading.CMD_PATH, cmd_button_queue.SEE_CLOTHING_INFO_ASK_PANEL
    ).copy()
    if wear_clothing_judge:
        del cmd_data["1"]
    else:
        del cmd_data["2"]
    cmd_list = list(cmd_data.values())
    return cmd_button_queue.option_int(
        None, 4, cmd_size="center", cmd_list_data=cmd_list
    )
