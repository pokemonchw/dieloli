from Script.Core import (
    py_cmd,
    text_loading,
    game_init,
    era_print,
    text_handle,
    game_config,
    constant,
)


def option_int(
    cmd_list: str,
    cmd_column=1,
    id_size="left",
    id_switch=True,
    askfor=True,
    cmd_size="left",
    start_id=0,
    cmd_list_data=None,
    last_line=False,
    normal_style_data={},
    on_style_data={},
) -> list:
    """
    批量绘制带id命令列表
    例:
    [000]开始游戏
    Keyword arguments:
    cmd_list -- 命令列表id，当cmd_list_data为None时，根据此id调用cmd_list内的命令数据
    cmd_column -- 每行命令列数 (default 1)
    id_size -- id文本位置(left/center/right) (default 'left')
    id_switch -- id显示开关 (default True)
    askfor -- 绘制完成时等待玩家输入的开关 (default True)
    cmd_size -- 命令文本在当前列的对齐方式(left/center/right) (default 'left')
    start_id -- 命令列表的起始id (default 0)
    cmd_list_data -- 命令列表数据 (default None)
    last_line -- 最后一个命令换行绘制 (default False)
    normal_style_data -- 按钮对应通常样式列表
    on_style_data -- 按钮对应按下时样式列表
    """
    if cmd_list_data is None:
        cmd_list_data = text_loading.get_text_data(constant.FilePath.CMD_PATH, cmd_list).copy()
    input_i = []
    text_width = game_config.text_width
    if last_line:
        if len(cmd_list_data) < cmd_column:
            cmd_column = len(cmd_list_data)
    else:
        if len(cmd_list_data) + 1 < cmd_column:
            cmd_column = len(cmd_list_data)
    cmd_index = int(text_width / cmd_column)
    if len(cmd_list_data) + 1 < cmd_column:
        cmd_column = len(cmd_list_data) + 1
    for i in range(0, len(cmd_list_data)):
        cmd_text = cmd_list_data[i]
        return_id = i + start_id
        if id_switch:
            id = id_index(return_id)
        else:
            id = ""
        cmd_text_and_id = id + cmd_text
        cmd_text_and_id_index = text_handle.get_text_index(cmd_text_and_id)
        normal_style = "standard"
        on_style = "onbutton"
        if cmd_list_data[i] in normal_style_data:
            normal_style = normal_style_data[cmd_list_data[i]]
        if cmd_list_data[i] in on_style_data:
            on_style = on_style_data[cmd_list_data[i]]
        if cmd_text_and_id_index < cmd_index:
            if id_size == "right":
                cmd_text_and_id = cmd_text + id
            elif id_size == "left":
                cmd_text_and_id = id + cmd_text
            if i == 0:
                cmd_text_and_id = cmd_text_and_id.rstrip()
                cmd_size_print(
                    cmd_text_and_id, return_id, None, cmd_index, cmd_size, True, normal_style, on_style,
                )
                input_i.append(str(return_id))
            elif i / cmd_column >= 1 and i % cmd_column == 0:
                era_print.line_feed_print()
                cmd_text_and_id = cmd_text_and_id.rstrip()
                cmd_size_print(
                    cmd_text_and_id, return_id, None, cmd_index, cmd_size, True, normal_style, on_style,
                )
                input_i.append(str(return_id))
            elif i == len(cmd_list_data) and last_line:
                era_print.line_feed_print()
                cmd_text_and_id = cmd_text_and_id.rstrip()
                cmd_size_print(
                    cmd_text_and_id, return_id, None, cmd_index, cmd_size, True, normal_style, on_style,
                )
                input_i.append(str(return_id))
            else:
                cmd_text_and_id = cmd_text_and_id.rstrip()
                cmd_size_print(
                    cmd_text_and_id, return_id, None, cmd_index, cmd_size, True, normal_style, on_style,
                )
                input_i.append(str(return_id))
    era_print.line_feed_print()
    if askfor:
        ans = int(game_init.askfor_int(input_i))
        return ans
    else:
        return input_i


def option_str(
    cmd_list: str,
    cmd_column=1,
    cmd_size="left",
    last_line=False,
    askfor=True,
    cmd_list_data=None,
    null_cmd="",
    return_data=None,
    normal_style_data={},
    on_style_data={},
    fix_cmd=True,
) -> list:
    """
    绘制无id的文本命令列表
    例:
    [长寿的青蛙]
    Keyword arguments:
    cmd_list -- 命令列表id，当cmd_list_data为None时，根据此id调用cmd_list内的命令数据
    cmd_column -- 每行命令列数 (default 1)
    cmd_size -- 命令文本在当前列的对齐方式(left/center/right) (default 'left')
    last_line -- 最后一个命令换行绘制 (default False)
    cmd_list_data -- 命令列表数据 (default None)
    null_cmd -- 在列表中按纯文本绘制，并不加入监听列表的命令文本
    return_data -- 命令返回数据 (default None)
    normal_style_data -- 按钮通常样式列表
    on_style_data -- 按钮被按下时样式列表
    """
    if cmd_list_data is None:
        cmd_list_data = text_loading.get_text_data(constant.FilePath.CMD_PATH, cmd_list).copy()
    input_s = []
    text_width = game_config.text_width
    if last_line:
        if len(cmd_list_data) - 1 < cmd_column:
            cmd_column = len(cmd_list_data) - 1
    else:
        if len(cmd_list_data) < cmd_column:
            cmd_column = len(cmd_list_data)
    cmd_index = int(text_width / cmd_column)
    now_null_cmd = null_cmd
    for i in range(0, len(cmd_list_data)):
        normal_style = "standard"
        on_style = "onbutton"
        if cmd_list_data[i] in normal_style_data:
            normal_style = normal_style_data[cmd_list_data[i]]
        if cmd_list_data[i] in on_style_data:
            on_style = on_style_data[cmd_list_data[i]]
        now_null_cmd = True
        if return_data is None:
            if null_cmd == cmd_list_data[i]:
                now_null_cmd = False
            cmd_text_bak = cmd_list_data[i]
            if fix_cmd:
                cmd_text = "[" + cmd_text_bak + "]"
            else:
                cmd_text = cmd_text_bak
        else:
            if null_cmd == return_data[i]:
                now_null_cmd = False
            cmd_text_bak = return_data[i]
            if fix_cmd:
                cmd_text = "[" + cmd_list_data[i] + "]"
            else:
                cmd_text = cmd_list_data[i]
        if i == 0:
            cmd_size_print(
                cmd_text, cmd_text_bak, None, cmd_index, cmd_size, now_null_cmd, normal_style, on_style,
            )
            if now_null_cmd:
                input_s.append(cmd_text_bak)
        elif i / cmd_column >= 1 and i % cmd_column == 0:
            era_print.line_feed_print()
            cmd_size_print(
                cmd_text, cmd_text_bak, None, cmd_index, cmd_size, now_null_cmd, normal_style, on_style,
            )
            if now_null_cmd:
                input_s.append(cmd_text_bak)
        elif i == len(cmd_list_data) - 1 and last_line:
            era_print.line_feed_print()
            cmd_size_print(
                cmd_text, cmd_text_bak, None, cmd_index, cmd_size, now_null_cmd, normal_style, on_style,
            )
            if now_null_cmd:
                input_s.append(cmd_text_bak)
        else:
            cmd_size_print(
                cmd_text, cmd_text_bak, None, cmd_index, cmd_size, now_null_cmd, normal_style, on_style,
            )
            if now_null_cmd:
                input_s.append(cmd_text_bak)
    era_print.line_feed_print()
    if askfor:
        ans = game_init.askfor_all(input_s)
        return ans
    else:
        return input_s


def id_index(id):
    """
    生成命令id文本
    Keyword arguments:
    id -- 命令id
    """
    if int(id) - 100 >= 0:
        id_s = "[" + str(id) + "] "
        return id_s
    elif int(id) - 10 >= 0:
        if int(id) == 0:
            id_s = "[00" + str(id) + "] "
            return id_s
        else:
            id_s = "[0" + str(id) + "] "
            return id_s
    else:
        id_s = "[00" + str(id) + "] "
        return id_s


def cmd_size_print(
    cmd_text,
    cmd_text_bak,
    cmd_event=None,
    text_width=0,
    cmd_size="left",
    no_null_cmd=True,
    normal_style="standard",
    on_style="onbutton",
):
    """
    计算命令对齐方式，补全文本并绘制
    Keyword arguments:
    cmd_text -- 命令文本
    cmd_text_bak -- 命令被触发时返回的文本
    cmd_event -- 命令绑定的事件 (default None)
    text_width -- 文本对齐时补全空间宽度
    cmd_size -- 命令对齐方式(left/center/right) (default 'left')
    no_null_cmd -- 绘制命令而非null命令样式的文本 (default False)
    normal_style -- 按钮通常样式 (default 'standard')
    on_style -- 按钮被按下时样式 (default 'onbutton')
    """
    if not no_null_cmd:
        cmd_text = "<nullcmd>" + cmd_text + "</nullcmd>"
    if cmd_size == "left":
        cmd_width = text_handle.get_text_index(cmd_text)
        cmd_text_fix = " " * (text_width - cmd_width)
        if no_null_cmd:
            py_cmd.pcmd(
                cmd_text, cmd_text_bak, cmd_event, normal_style=normal_style, on_style=on_style,
            )
        else:
            era_print.normal_print(cmd_text)
        era_print.normal_print(cmd_text_fix)
    elif cmd_size == "center":
        cmd_width = text_handle.get_text_index(cmd_text)
        cmd_text_fix = " " * (int(text_width / 2) - int(cmd_width / 2))
        era_print.normal_print(cmd_text_fix)
        if no_null_cmd:
            py_cmd.pcmd(
                cmd_text, cmd_text_bak, cmd_event, normal_style=normal_style, on_style=on_style,
            )
        else:
            era_print.normal_print(cmd_text)
        era_print.normal_print(cmd_text_fix)
    elif cmd_size == "right":
        cmd_width = text_handle.get_text_index(cmd_text)
        cmd_text_fix = " " * (text_width - cmd_width)
        if no_null_cmd:
            py_cmd.pcmd(
                cmd_text, cmd_text_bak, cmd_event, normal_style=normal_style, on_style=on_style,
            )
        else:
            era_print.normal_print(cmd_text)
        era_print.normal_print(cmd_text_fix)
