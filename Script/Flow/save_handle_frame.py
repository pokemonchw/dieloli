from Script.Core import (
    game_config,
    cache_contorl,
    game_init,
    py_cmd,
    save_handle,
)
from Script.Panel import save_handle_frame_panel


def establish_save_func():
    """
    绘制保存存档界面流程
    """
    while True:
        input_s = []
        save_page = save_page_index()
        show_save_value = save_page[0]
        last_save_page_value = save_page[1]
        save_handle_frame_panel.establish_save_info_head_panel()
        flow_return = save_handle_frame_panel.see_save_list_panel(
            show_save_value, last_save_page_value
        )
        input_s = input_s + flow_return
        start_id = len(input_s)
        flow_return = save_handle_frame_panel.ask_for_change_save_page_panel(
            start_id
        )
        input_s = input_s + flow_return
        yrn = game_init.askfor_int(input_s)
        py_cmd.clr_cmd()
        if yrn == str(start_id):
            save_panel_page = int(
                cache_contorl.panel_state["SeeSaveListPanel"]
            )
            if save_panel_page == 0:
                cache_contorl.panel_state[
                    "SeeSaveListPanel"
                ] = cache_contorl.max_save_page
            else:
                cache_contorl.panel_state["SeeSaveListPanel"] = (
                    save_panel_page - 1
                )
        elif yrn == str(start_id + 1):
            cache_contorl.panel_state["SeeSaveListPanel"] = 0
            cache_contorl.now_flow_id = cache_contorl.old_flow_id
            break
        elif yrn == str(start_id + 2):
            save_panel_page = int(
                cache_contorl.panel_state["SeeSaveListPanel"]
            )
            if save_panel_page == cache_contorl.max_save_page:
                cache_contorl.panel_state["SeeSaveListPanel"] = 0
            else:
                cache_contorl.panel_state["SeeSaveListPanel"] = (
                    save_panel_page + 1
                )
        else:
            ans_return = int(yrn)
            save_id = str(
                save_handle.get_save_page_save_id(show_save_value, ans_return)
            )
            if save_handle.judge_save_file_exist(save_id):
                ask_for_overlay_save_func(save_id)
            else:
                save_handle.establish_save(save_id)


def load_save_func():
    """
    绘制读取存档界面流程
    """
    while True:
        input_s = []
        save_page = save_page_index()
        show_save_value = save_page[0]
        last_save_page_value = save_page[1]
        save_handle_frame_panel.load_save_info_head_panel()
        flow_return = save_handle_frame_panel.see_save_list_panel(
            show_save_value, last_save_page_value, True
        )
        input_s = input_s + flow_return
        start_id = len(input_s)
        flow_return = save_handle_frame_panel.ask_for_change_save_page_panel(
            start_id
        )
        input_s = input_s + flow_return
        yrn = game_init.askfor_int(input_s)
        py_cmd.clr_cmd()
        if yrn == str(start_id):
            save_panel_page = int(
                cache_contorl.panel_state["SeeSaveListPanel"]
            )
            if save_panel_page == 0:
                cache_contorl.panel_state[
                    "SeeSaveListPanel"
                ] = cache_contorl.max_save_page
            else:
                cache_contorl.panel_state["SeeSaveListPanel"] = (
                    save_panel_page - 1
                )
        elif yrn == str(start_id + 1):
            cache_contorl.panel_state["SeeSaveListPanel"] = 0
            cache_contorl.now_flow_id = cache_contorl.old_flow_id
            break
        elif yrn == str(start_id + 2):
            save_panel_page = int(
                cache_contorl.panel_state["SeeSaveListPanel"]
            )
            if save_panel_page == cache_contorl.max_save_page:
                cache_contorl.panel_state["SeeSaveListPanel"] = 0
            else:
                cache_contorl.panel_state["SeeSaveListPanel"] = (
                    save_panel_page + 1
                )
        else:
            ans_return = int(yrn)
            save_id = save_handle.get_save_page_save_id(
                show_save_value, ans_return
            )
            if ask_for_load_save_func(str(save_id)):
                break


def save_page_index():
    """
    用于计算存档页面单页存档显示数量
    return:
    save_page[0] -- 存档页面单页显示数量
    save_page[1] -- 最大存档数不能被存档页数整除时，额外存档页存档数量
    """
    max_save_value = int(game_config.max_save)
    page_save_value = int(game_config.save_page)
    last_save_page_value = 0
    if max_save_value % page_save_value != 0:
        show_save_value = int(max_save_value / page_save_value)
        last_save_page_value = max_save_value % page_save_value
        cache_contorl.max_save_page = page_save_value
    else:
        cache_contorl.max_save_page = page_save_value - 1
        show_save_value = max_save_value / page_save_value
    save_page = [show_save_value, last_save_page_value]
    return save_page


def ask_for_overlay_save_func(save_id: str):
    """
    存档处理询问流程
    玩家输入0:进入覆盖存档询问流程
    玩家输入1:进入删除存档询问流程
    Keyword arguments:
    save_id -- 存档id
    """
    cmd_list = save_handle_frame_panel.ask_for_overlay_save_panel()
    yrn = game_init.askfor_all(cmd_list)
    yrn = str(yrn)
    py_cmd.clr_cmd()
    if yrn == "0":
        confirmation_overlay_save_func(save_id)
    elif yrn == "1":
        confirmation_remove_save_func(save_id)


def confirmation_overlay_save_func(save_id: str):
    """
    覆盖存档询问流程
    玩家输入0:对存档进行覆盖
    Keyword arguments:
    save_id -- 存档id
    """
    cmd_list = save_handle_frame_panel.confirmation_overlay_save_panel()
    yrn = game_init.askfor_all(cmd_list)
    py_cmd.clr_cmd()
    if yrn == "0":
        save_handle.establish_save(save_id)


def ask_for_load_save_func(save_id: str):
    """
    读档处理询问流程
    玩家输入0:进入读取存档询问流程
    玩家输入1:进入删除存档询问流程
    Keyword arguments:
    save_id -- 存档id
    """
    cmd_list = save_handle_frame_panel.ask_load_save_panel()
    yrn = game_init.askfor_all(cmd_list)
    py_cmd.clr_cmd()
    if yrn == "0":
        return confirmation_load_save_func(save_id)
    elif yrn == "1":
        confirmation_remove_save_func(save_id)
    return False


def confirmation_load_save_func(save_id: str):
    """
    读取存档询问流程
    玩家输入0:读取指定存档
    Keyword arguments:
    save_id -- 存档id
    """
    cmd_list = save_handle_frame_panel.confirmation_load_save_panel()
    yrn = game_init.askfor_all(cmd_list)
    py_cmd.clr_cmd()
    if yrn == "0":
        save_handle.input_load_save(save_id)
        cache_contorl.now_flow_id = "main"
        return True
    return False


def confirmation_remove_save_func(save_id: str):
    """
    覆盖存档询问流程
    玩家输入0:删除指定存档
    Keyword arguments:
    save_id -- 存档id
    """
    cmd_list = save_handle_frame_panel.confirmation_remove_save_panel()
    yrn = game_init.askfor_all(cmd_list)
    if yrn == "0":
        save_handle.remove_save(save_id)
    py_cmd.clr_cmd()
