from Script.Core import cache_contorl, text_loading, era_print, py_cmd
from Script.Design import cmd_button_queue


def see_instruct_head_panel() -> list:
    """
    绘制指令面板的头部过滤器面板
    Return arguments:
    list -- 绘制的按钮列表
    """
    era_print.little_title_print(
        text_loading.get_text_data(text_loading.STAGE_WORD_PATH, "146")
    )
    instruct_data = text_loading.get_text_data(
        text_loading.CMD_PATH, cmd_button_queue.INSTRUCT_HEAD_PANEL
    )
    if cache_contorl.instruct_filter == {}:
        cache_contorl.instruct_filter = {instruct: 0 for instruct in instruct_data}
        cache_contorl.instruct_filter["Dialogue"] = 1
    style_data = {
        instruct_data[instruct]: "selectmenu"
        for instruct in instruct_data
        if cache_contorl.instruct_filter[instruct] == 0
    }
    on_style_data = {
        instruct_data[instruct]: "onselectmenu"
        for instruct in instruct_data
        if cache_contorl.instruct_filter[instruct] == 0
    }
    era_print.normal_print(
        text_loading.get_text_data(text_loading.STAGE_WORD_PATH, "147")
    )
    return cmd_button_queue.option_str(
        None,
        len(instruct_data),
        "center",
        False,
        False,
        list(instruct_data.values()),
        "",
        list(instruct_data.keys()),
        style_data,
        on_style_data,
    )
