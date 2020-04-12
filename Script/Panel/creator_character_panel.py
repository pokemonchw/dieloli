from Script.Core import (
    cache_contorl,
    era_print,
    text_loading,
    py_cmd,
    game_init,
    text_handle,
)
from Script.Design import attr_calculation, cmd_button_queue


def input_name_panel() -> str:
    """
    请求玩家输入姓名面板
    """
    era_print.line_feed_print(
        text_loading.get_text_data(text_loading.MESSAGE_PATH, "4")
    )
    yrn = cmd_button_queue.option_int(cmd_button_queue.CURRENCY_MENU)
    return yrn


def start_input_name_panel():
    """
    玩家姓名输入处理面板
    """
    era_print.line_feed_print(
        text_loading.get_text_data(text_loading.MESSAGE_PATH, "3")
    )
    while 1:
        character_name = game_init.askfor_str()
        era_print.line_feed_print(character_name)
        if text_handle.get_text_index(character_name) > 10:
            era_print.line_feed_print(
                text_loading.get_text_data(
                    text_loading.ERROR_PATH, "inputNameTooLongError"
                )
            )
        else:
            cache_contorl.character_data["character"][0].name = character_name
            break


def input_nick_name_panel() -> str:
    """
    请求玩家输入昵称面板
    """
    era_print.line_feed_print(
        text_loading.get_text_data(text_loading.MESSAGE_PATH, "6")
    )
    yrn = cmd_button_queue.option_int(cmd_button_queue.INPUT_NICK_NAME)
    return yrn


def start_input_nick_name_panel():
    """
    玩家昵称输入处理面板
    """
    era_print.line_feed_print(
        text_loading.get_text_data(text_loading.MESSAGE_PATH, "5")
    )
    while 1:
        character_nick_name = game_init.askfor_str()
        era_print.line_feed_print(character_nick_name)
        if text_handle.get_text_index(character_nick_name) > 10:
            era_print.line_feed_print(
                text_loading.get_text_data(
                    text_loading.ERROR_PATH, "inputNickNameTooLongError"
                )
            )
        else:
            cache_contorl.character_data["character"][0].nick_name = character_nick_name
            break


def input_self_name_panel() -> str:
    """
    请求玩家输入自称面板
    """
    py_cmd.clr_cmd()
    era_print.line_feed_print(
        text_loading.get_text_data(text_loading.MESSAGE_PATH, "14")
    )
    yrn = cmd_button_queue.option_int(cmd_button_queue.INPUT_SELF_NEME)
    return yrn


def start_input_self_name():
    """
    玩家自称输入处理面板
    """
    era_print.line_feed_print()
    era_print.line_feed_print(
        text_loading.get_text_data(text_loading.MESSAGE_PATH, "15")
    )
    while 1:
        character_self_name = game_init.askfor_str()
        era_print.line_feed_print(character_self_name)
        if text_handle.get_text_index(character_self_name) > 10:
            era_print.line_feed_print(
                text_loading.get_text_data(
                    text_loading.ERROR_PATH, "inputSelfNameTooLongError"
                )
            )
        else:
            cache_contorl.character_data["character"][0].self_name = character_self_name
            break


def input_sex_panel() -> str:
    """
    请求玩家选择性别面板
    """
    character_id = cache_contorl.character_data["character_id"]
    sex_id = cache_contorl.character_data["character"][character_id].sex
    era_print.line_feed_print(
        text_loading.get_text_data(text_loading.MESSAGE_PATH, "8")[sex_id]
    )
    yrn = cmd_button_queue.option_int(cmd_button_queue.CURRENCY_MENU)
    return yrn


def input_sex_choice_panel() -> str:
    """
    玩家性别选择面板
    """
    era_print.line_feed_print(
        text_loading.get_text_data(text_loading.MESSAGE_PATH, "7")
    )
    yrn = cmd_button_queue.option_int(cmd_button_queue.SEX_MENU)
    return yrn


def attribute_generation_branch_panel() -> str:
    """
    玩家确认进行详细设置面板
    """
    py_cmd.clr_cmd()
    era_print.line_feed_print(
        text_loading.get_text_data(text_loading.MESSAGE_PATH, "9")
    )
    yrn = cmd_button_queue.option_int(cmd_button_queue.CURRENCY_MENU)
    return yrn


def detailed_setting_1_panel() -> str:
    """
    询问玩家年龄模板面板
    """
    era_print.line_feed_print(
        text_loading.get_text_data(text_loading.MESSAGE_PATH, "10")
    )
    yrn = cmd_button_queue.option_int(cmd_button_queue.DETAILED_SETTING1)
    return yrn


def detailed_setting_3_panel() -> str:
    """
    询问玩家性经验程度面板
    """
    era_print.line_feed_print(
        text_loading.get_text_data(text_loading.MESSAGE_PATH, "12")
    )
    yrn = cmd_button_queue.option_int(cmd_button_queue.DETAILED_SETTING3)
    return yrn


def detailed_setting_8_panel() -> str:
    """
    询问玩家肥胖程度面板
    """
    era_print.line_feed_print(
        text_loading.get_text_data(text_loading.MESSAGE_PATH, "29")
    )
    yrn = cmd_button_queue.option_int(cmd_button_queue.DETAILED_SETTING8)
    return yrn


def enter_character_nature_head():
    """
    用于确认角色性格的头部面板
    """
    era_print.line_feed_print(
        text_loading.get_text_data(text_loading.MESSAGE_PATH, "39")
    )


def enter_character_nature_end() -> list:
    """
    用户确认角色性格的尾部面板
    Return arguments:
    list -- 按钮列表
    """
    era_print.line_feed_print()
    return cmd_button_queue.option_int(
        cmd_button_queue.ENTER_CHARACTER_NATURE, 1, "left", True, False
    )
