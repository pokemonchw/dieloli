from Script.Core import era_print, cache_contorl, text_loading, py_cmd
from Script.Design import attr_text, attr_print, game_time, cmd_button_queue


def main_frame_panel() -> list:
    """
    游戏主菜单
    """
    cmd_list = []
    character_id = cache_contorl.character_data["character_id"]
    character_data = cache_contorl.character_data["character"][character_id]
    title_text = text_loading.get_text_data(text_loading.STAGE_WORD_PATH, "64")
    era_print.little_title_print(title_text)
    date_text = game_time.get_date_text()
    era_print.normal_print(date_text)
    era_print.normal_print(" ")
    week_date_text = game_time.get_week_day_text()
    era_print.normal_print(week_date_text)
    era_print.normal_print(" ")
    character_name = character_data.name
    py_cmd.pcmd(character_name, character_name, None)
    cmd_list.append(character_name)
    era_print.normal_print(" ")
    gold_text = attr_text.get_gold_text(character_id)
    era_print.line_feed_print(gold_text)
    attr_print.print_hp_and_mp_bar(character_id)
    main_menu_text = text_loading.get_text_data(text_loading.STAGE_WORD_PATH, "68")
    era_print.son_title_print(main_menu_text)
    era_print.line_feed_print("\n")
    ask_for_main_menu = cmd_button_queue.option_int(
        cmd_button_queue.MAIN_MENU, 3, "left", askfor=False, cmd_size="center"
    )
    cmd_list = cmd_list + ask_for_main_menu
    system_menu_text = text_loading.get_text_data(text_loading.STAGE_WORD_PATH, "69")
    era_print.son_title_print(system_menu_text)
    era_print.line_feed_print()
    system_menu_start_id = len(ask_for_main_menu)
    ask_for_system_menu = cmd_button_queue.option_int(
        cmd_button_queue.SYSTEM_MENU,
        4,
        "left",
        askfor=False,
        cmd_size="center",
        start_id=system_menu_start_id,
    )
    cmd_list = cmd_list + ask_for_system_menu
    return cmd_list
