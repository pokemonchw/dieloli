from Script.Core import (
    cache_contorl,
    game_config,
    py_cmd,
    text_loading,
    era_print,
    text_handle,
    save_handle,
    constant,
)
from Script.Design import cmd_button_queue, game_time


def load_save_info_head_panel():
    """
    载入存档信息头面板
    """
    save_frame_title = text_loading.get_text_data(
        constant.FilePath.STAGE_WORD_PATH, "71"
    )
    era_print.little_title_print(save_frame_title)


def establish_save_info_head_panel():
    """
    存储存档信息头面板
    """
    save_frame_title = text_loading.get_text_data(
        constant.FilePath.STAGE_WORD_PATH, "70"
    )
    era_print.little_title_print(save_frame_title)


def see_save_list_panel(
    page_save_value: int, last_save_page_value: int, auto_save=False
) -> list:
    """
    查看存档页面面板
    Keyword arguments:
    page_save_value -- 单页最大存档显示数量
    last_save_page_value -- 最后一页存档显示数量
    auto_save -- 自动存档显示开关 (default False)
    """
    save_panel_page = int(cache_contorl.panel_state["SeeSaveListPanel"]) + 1
    input_s = []
    id_text_list = []
    id_info_text = text_loading.get_text_data(
        constant.FilePath.STAGE_WORD_PATH, "72"
    )
    text_width = int(game_config.text_width)
    save_none_text = text_loading.get_text_data(
        constant.FilePath.MESSAGE_PATH, "20"
    )
    if save_panel_page == int(game_config.save_page) + 1:
        start_save_id = int(page_save_value) * (save_panel_page - 1)
        over_save_id = start_save_id + last_save_page_value
    else:
        over_save_id = int(page_save_value) * save_panel_page
        start_save_id = over_save_id - int(page_save_value)
    for i in range(0, over_save_id - start_save_id):
        id = cmd_button_queue.id_index(i)
        save_id = start_save_id + i
        if auto_save and not save_handle.judge_save_file_exist(str(save_id)):
            id_text = id_info_text + " " + str(save_id) + ":"
            id_text_list.append(id_text)
        else:
            id_text = id + id_info_text + " " + str(save_id) + ":"
            id_text_list.append(id_text)
    for i in range(0, over_save_id - start_save_id):
        id = str(i)
        id_text = id_text_list[i]
        era_print.little_line_print()
        save_id = str(save_handle.get_save_page_save_id(page_save_value, i))
        if save_handle.judge_save_file_exist(save_id):
            save_info_head = save_handle.load_save_info_head(save_id)
            game_time_data = save_info_head["game_time"]
            game_time_text = game_time.get_date_text(game_time_data)
            character_name = save_info_head["character_name"]
            save_verson = save_info_head["game_verson"]
            save_text = (
                character_name + " " + game_time_text + " " + save_verson
            )
            id_text_index = int(text_handle.get_text_index(id_text))
            fix_id_width = text_width - id_text_index
            save_align = text_handle.align(
                save_text, "center", text_width=fix_id_width
            )
            id_text = id_text + save_align
            py_cmd.pcmd(id_text, id, None)
            input_s.append(id)
        else:
            id_text_index = int(text_handle.get_text_index(id_text))
            fix_id_width = text_width - id_text_index
            save_none_align = text_handle.align(
                save_none_text, "center", text_width=fix_id_width
            )
            id_text = id_text + save_none_align
            if auto_save:
                era_print.normal_print(id_text)
            else:
                py_cmd.pcmd(id_text, id, None)
                input_s.append(id)
        era_print.line_feed_print()
    if auto_save:
        auto_info_text = text_loading.get_text_data(
            constant.FilePath.STAGE_WORD_PATH, "73"
        )
        i = page_save_value
        id = cmd_button_queue.id_index(i)
        era_print.little_line_print()
        if save_handle.judge_save_file_exist("auto"):
            save_info_head = save_handle.load_save_info_head("auto")
            game_time_data = save_info_head["game_time"]
            game_time_text = game_time.get_date_text(game_time_data)
            character_name = save_info_head["character_name"]
            save_verson = save_info_head["game_verson"]
            save_text = (
                character_name + " " + game_time_text + " " + save_verson
            )
            id_text = id + auto_info_text
            id_text_index = int(text_handle.get_text_index(id_text))
            fix_id_width = text_width - id_text_index
            save_text_align = text_handle.align(
                save_text, "center", text_width=fix_id_width
            )
            id_text = id_text + save_text_align
            py_cmd.pcmd(id_text, id, None)
            input_s.append(id)
            era_print.line_feed_print()
        else:
            id_text_index = int(text_handle.get_text_index(auto_info_text))
            fix_id_width = text_width - id_text_index
            save_none_align = text_handle.align(
                save_none_text, "center", text_width=fix_id_width
            )
            id_text = auto_info_text + save_none_align
            era_print.normal_print(id_text)
            era_print.line_feed_print()
    return input_s


def ask_for_change_save_page_panel(start_id: str) -> list:
    """
    询问切换存档页面面板
    Keyword arguments:
    start_id -- 面板命令的起始id
    """
    cmd_list = text_loading.get_text_data(
        constant.FilePath.CMD_PATH, "changeSavePage"
    )
    save_panel_page = str(cache_contorl.panel_state["SeeSaveListPanel"])
    max_save_panel_page = str(cache_contorl.max_save_page)
    save_page_text = "(" + save_panel_page + "/" + max_save_panel_page + ")"
    era_print.page_line_print(sample="-", string=save_page_text)
    era_print.line_feed_print()
    yrn = cmd_button_queue.option_int(
        None,
        3,
        askfor=False,
        cmd_size="center",
        start_id=start_id,
        cmd_list_data=cmd_list,
    )
    return yrn


def ask_for_overlay_save_panel() -> list:
    """
    询问覆盖存档面板
    """
    era_print.line_feed_print()
    cmd_list = text_loading.get_text_data(
        constant.FilePath.CMD_PATH, "overlay_save"
    )
    message_text = text_loading.get_text_data(
        constant.FilePath.MESSAGE_PATH, "21"
    )
    era_print.restart_line_print()
    era_print.normal_print(message_text)
    era_print.line_feed_print()
    yrn = cmd_button_queue.option_int(
        None, askfor=False, cmd_list_data=cmd_list
    )
    return yrn


def confirmation_overlay_save_panel() -> list:
    """
    确认覆盖存档面板
    """
    era_print.line_feed_print()
    cmd_list = text_loading.get_text_data(
        constant.FilePath.CMD_PATH, "confirmation_overlay_save"
    )
    message_text = text_loading.get_text_data(
        constant.FilePath.MESSAGE_PATH, "22"
    )
    era_print.restart_line_print()
    era_print.line_feed_print(message_text)
    yrn = cmd_button_queue.option_int(
        None, askfor=False, cmd_list_data=cmd_list
    )
    return yrn


def ask_load_save_panel() -> list:
    """
    询问读取存档面板
    """
    era_print.line_feed_print()
    cmd_list = text_loading.get_text_data(
        constant.FilePath.CMD_PATH, "loadSaveAsk"
    )
    message_text = text_loading.get_text_data(
        constant.FilePath.MESSAGE_PATH, "23"
    )
    era_print.restart_line_print()
    era_print.line_feed_print(message_text)
    yrn = cmd_button_queue.option_int(
        None, askfor=False, cmd_list_data=cmd_list
    )
    return yrn


def confirmation_load_save_panel() -> list:
    """
    确认读取存档面板
    """
    era_print.line_feed_print()
    cmd_list = text_loading.get_text_data(
        constant.FilePath.CMD_PATH, "confirmationLoadSave"
    )
    message_text = text_loading.get_text_data(
        constant.FilePath.MESSAGE_PATH, "24"
    )
    era_print.restart_line_print()
    era_print.line_feed_print(message_text)
    yrn = cmd_button_queue.option_int(
        None, askfor=False, cmd_list_data=cmd_list
    )
    return yrn


def confirmation_remove_save_panel() -> list:
    """
    确认删除存档面板
    """
    era_print.line_feed_print()
    cmd_list = text_loading.get_text_data(
        constant.FilePath.CMD_PATH, "confirmationRemoveSave"
    )
    message_text = text_loading.get_text_data(
        constant.FilePath.MESSAGE_PATH, "25"
    )
    era_print.restart_line_print()
    era_print.line_feed_print(message_text)
    yrn = cmd_button_queue.option_int(
        None, askfor=False, cmd_list_data=cmd_list
    )
    return yrn
