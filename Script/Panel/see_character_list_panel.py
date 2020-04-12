from Script.Core import (
    text_loading,
    era_print,
    cache_contorl,
    game_config,
    py_cmd,
    text_handle,
)
from Script.Design import character_handle, cmd_button_queue, attr_text


def see_character_list_panel(max_page: int) -> list:
    """
    查看角色列表面板
    Keyword arguments:
    max_page -- 最大角色列表页数
    """
    title_text = text_loading.get_text_data(text_loading.STAGE_WORD_PATH, "74")
    era_print.little_title_print(title_text)
    input_s = []
    page_id = int(cache_contorl.panel_state["SeeCharacterListPanel"])
    page_show = int(game_config.character_list_show)
    character_max = len(cache_contorl.character_data["character"]) - 1
    if page_id == max_page:
        show_page_start = page_show * page_id
        show_page_over = show_page_start + (character_max + 1 - show_page_start)
    else:
        show_page_over = page_show * (page_id + 1)
        show_page_start = show_page_over - page_show
    for i in range(show_page_start, show_page_over):
        cmd_id = i - show_page_start
        cmd_id_text = cmd_button_queue.id_index(cmd_id)
        cmd_text = attr_text.get_character_abbreviations_info(i)
        cmd_id_text_index = text_handle.get_text_index(cmd_id_text)
        window_width = int(game_config.text_width)
        text_width = window_width - cmd_id_text_index
        cmd_text = text_handle.align(cmd_text, "center", text_width=text_width)
        cmd_text = cmd_id_text + " " + cmd_text
        cmd_id = str(cmd_id)
        era_print.little_line_print()
        py_cmd.pcmd(cmd_text, cmd_id, None)
        input_s.append(cmd_id)
        era_print.normal_print('\n')
    page_text = "(" + str(page_id) + "/" + str(max_page) + ")"
    era_print.page_line_print(sample="-", string=page_text)
    era_print.line_feed_print()
    return input_s


def ask_for_see_character_list_panel(start_id: str) -> list:
    """
    切换角色列表页面处理面板
    Keyword arguments:
    start_id -- 面板命令起始id
    """
    yrn = cmd_button_queue.option_int(
        cmd_button_queue.SEE_CHARACTER_LIST,
        3,
        "left",
        askfor=False,
        cmd_size="center",
        start_id=start_id,
    )
    return yrn
