import time
from Script.Core import (
    text_handle,
    flow_handle,
    io_init,
    cache_contorl,
    dictionaries,
    rich_text,
    constant,
)
from Script.Config import game_config

last_char = "\n"

# 默认输出样式
def_style = io_init.style_def


def normal_print(string: str, style="standard", rich_text_judge=True):
    """
    游戏基础的文本绘制实现
    Keyword arguments:
    string -- 需要绘制的文本
    style -- 文本的默认样式 (default 'standard')
    rich_text_judge -- 启用富文本的开关 (default True)
    """
    if rich_text_judge:
        string = dictionaries.handle_text(string)
        string.__repr__
        style_list = rich_text.get_rich_text_print(string, style)
        global last_char
        if len(string) > 0:
            last_char = string[-1:]
        string = rich_text.remove_rich_cache(string)
        string = r"" + string
        for i in range(0, len(string)):
            io_init.era_print(string[i], style_list[i])
    else:
        io_init.era_print(string, style)


def image_print(image_name: str, image_path=""):
    """
    图片绘制在era_print中的封装
    Keyword arguments:
    image_name -- 图片id
    image_path -- 图片所在路径 (default '')
    """
    io_init.image_print(image_name, image_path)


def little_title_print(string: str):
    """
    按预订样式"littletitle(小标题)"绘制文本
    示例:
    ====口小标题口====
    文本将用=补全至与行同宽
    Keyword arguments:
    string -- 小标题文本
    """
    text_wait = cache_contorl.text_wait
    if text_wait != 0:
        time.sleep(text_wait)
    string = str(string)
    string = dictionaries.handle_text(string)
    global last_char
    if len(string) > 0:
        last_char = string[-1:]
    width = game_config.config_normal.text_width
    text_width = text_handle.get_text_index(string)
    line_width = int(int(width) / 2 - int(text_width) / 2 - 2)
    line_feed_print("=" * line_width + "<littletitle>口" + string + "口</littletitle>" + "=" * line_width)


def son_title_print(string: str):
    """
    按预订样式"sontitle(子标题)"绘制文本
    示例：
    ::::子标题::::
    文本将用=补全至与行同宽
    Keyword arguments:
    string -- 子标题文本
    """
    text_wait = cache_contorl.text_wait
    if text_wait != 0:
        time.sleep(text_wait)
    string = string
    string = dictionaries.handle_text(string)
    global last_char
    if len(string) > 0:
        last_char = string[-1:]
    width = game_config.config_normal.text_width
    text_width = text_handle.get_text_index(string)
    line_width = int(int(width) / 4)
    line_width_fix = int(int(width) / 4 - int(text_width))
    line_feed_print(":" * line_width_fix + "<sontitle>" + string + "</sontitle>" + ":" * line_width * 3)


def line_feed_print(string="", style="standard"):
    """
    换行绘制文本并换行
    当已绘制文本末尾无换行符时,本次输出文本前新增换行符
    当本次输出文本末尾无换行符,本次输出末尾增加换行符
    Keyword arguments:
    string -- 要绘制的文本
    style -- 文本的默认样式 (default 'standard')
    """
    text_wait = cache_contorl.text_wait
    if text_wait != 0:
        time.sleep(text_wait)
    global last_char
    if not last_char == "\n":
        normal_print("\n")
    normal_print(str(string), style)
    if not last_char == "\n":
        normal_print("\n")


def restart_line_print(sample="=", style="standard"):
    """
    绘制一行指定文本
    Keyword arguments:
    string -- 要绘制的文本 (default '=')
    style -- 文本的默认样式 (default 'standard')
    """
    text_wait = cache_contorl.text_wait
    if text_wait != 0:
        time.sleep(text_wait)
    text_width = game_config.config_normal.text_width
    line_feed_print(sample * text_width, style)


def little_line_print():
    """
    绘制标题线，字符为':'
    """
    restart_line_print(":")


def page_line_print(sample=":", string="", style="standard"):
    """
    绘制页数线
    Keyword arguments:
    sample -- 填充线样式 (default ':')
    string -- 页数字符串 (default '')
    style -- 页数线默认样式 (default 'standard')
    """
    text_wait = cache_contorl.text_wait
    if text_wait != 0:
        time.sleep(text_wait)
    text_width = int(game_config.config_normal.text_width)
    string_width = int(text_handle.get_text_index(string))
    fix_text = sample * int(text_width / 2 - string_width / 2)
    string_text = fix_text + string + fix_text
    normal_print(string_text, style)


def warning_print(string: str, style="warning"):
    """
    绘制警告信息(将同时在终端打印)
    Keyword arguments:
    string -- 警告信息文本
    style -- 警告信息的默认样式 (default 'warning')
    """
    text_wait = cache_contorl.text_wait
    if text_wait != 0:
        time.sleep(text_wait)
    restart_line_print(string, style)
    print(string)


def wait_print(string: str, style="standard"):
    """
    绘制文本并等待玩家按下回车或鼠标左键
    Keyword arguments:
    string -- 要绘制的文本
    style -- 绘制文本的默认样式 (default 'standard')
    """
    text_wait = cache_contorl.text_wait
    if text_wait != 0:
        time.sleep(text_wait)
    normal_print(string, style)


def line_feed_wait_print(string="", style="standard"):
    """
    绘制文本换行并等待玩家按下回车或鼠标左键
    Keyword arguments:
    string -- 要绘制的文本 (default '')
    style -- 绘制文本的默认样式 (default 'standard')
    """
    text_wait = cache_contorl.text_wait
    if text_wait:
        time.sleep(text_wait)
    line_feed_print(string, style)
    cache_contorl.wframe_mouse.w_frame_up = 0
    flow_handle.askfor_wait()


def one_by_one_print(sleep_time: float, string: str, style="standard"):
    """
    逐字绘制文本
    Keyword arguments:
    sleep_time -- 逐字绘制时，绘制间隔时间
    string -- 需要逐字绘制的文本
    style -- 绘制文本的默认样式 (default 'standard')
    """
    text_wait = cache_contorl.text_wait
    if text_wait != 0:
        time.sleep(text_wait)
    cache_contorl.wframe_mouse.w_frame_up = 0
    style_list = rich_text.get_rich_text_print(string, style)
    style_name_list = list(game_config.config_font_data.keys())
    for i in range(0, len(style_name_list)):
        style_text_head = "<" + style_name_list[i] + ">"
        style_text_tail = "</" + style_name_list[i] + ">"
        if style_text_head in string:
            string = string.replace(style_text_head, "")
            string = string.replace(style_text_tail, "")
    index = len(string)
    for i in range(0, index):
        normal_print(string[i], style_list[i])
        time.sleep(sleep_time)
        if cache_contorl.wframe_mouse.w_frame_up:
            index_i = i + 1
            cache_contorl.wframe_mouse.w_frame_up = 2
            for index_i in range(index_i, index):
                normal_print(string[index_i], style_list[index_i])
            if cache_contorl.wframe_mouse.w_frame_lines_state == 2:
                cache_contorl.wframe_mouse.w_frame_lines_up = 2
            break


def list_print(string_list: list, string_column=1, string_size="left"):
    """
    绘制字符串列表
    Keyword arguments:
    string_list -- 要进行绘制的字符串列表
    string_colum -- 每行的绘制数量(列宽由行宽平分为行数而来) (default 1)
    string_size -- 每列在列宽中的对齐方式(left/center/right) (default 'left')
    """
    text_wait = cache_contorl.text_wait
    text_width = game_config.config_normal.text_width
    if text_wait != 0:
        time.sleep(text_wait)
    string_index = int(text_width / string_column)
    for i in range(0, len(string_list)):
        string_text = string_list[i]
        string_id_index = text_handle.get_text_index(string_list[i])
        if string_size == "left":
            string_text_fix = " " * (string_index - string_id_index)
            string_text = string_text + string_text_fix
        elif string_size == "center":
            string_text_fix = " " * int((string_index - string_id_index) / 2)
            string_text = string_text_fix + string_text + string_text_fix
            now_text_index = text_handle.get_text_index(string_text)
            if string_text_fix != "" and now_text_index < string_index:
                string_text += " " * (string_index - now_text_index)
            elif string_text_fix != "" and now_text_index > string_index:
                string_text = string_text[-1]
        elif string_size == "right":
            string_text_fix = " " * (string_index - string_id_index)
            string_text = string_text_fix + string_text
        string_text_index = text_handle.get_text_index(string_text)
        if string_text_index > string_index:
            string_text = string_text[:string_index]
        elif string_text_index < string_index:
            string_text = " " * (string_index - string_text_index) + string_text
        if i == 0:
            normal_print(string_text)
        elif i / string_column >= 1 and i % string_column == 0:
            normal_print("\n")
            normal_print(string_text)
        else:
            normal_print(string_text)


def next_screen_print():
    """
    绘制一整屏空行
    """
    text_wait = cache_contorl.text_wait
    if text_wait != 0:
        time.sleep(text_wait)
    normal_print("\n" * game_config.config_normal.text_hight)


def lines_center_print(sleep_time: float, string="", style="standard"):
    """
    将多行文本以居中的对齐方式进行逐字绘制
    Keyword arguments:
    sleep_time -- 逐字的间隔时间
    string -- 需要逐字绘制的文本
    style -- 文本的默认样式
    """
    text_wait = cache_contorl.text_wait
    if text_wait != 0:
        time.sleep(text_wait)
    cache_contorl.wframe_mouse.w_frame_lines_state = 1
    string = str(string)
    string_list = string.split("\n")
    width = game_config.config_normal.text_width
    style_name_list = list(game_config.config_font.keys())
    string_center_list = ""
    for i in range(0, len(style_name_list)):
        style_text_head = "<" + style_name_list[i] + ">"
        style_text_tail = "</" + style_name_list[i] + ">"
        if style_text_head in string:
            string_center = string.replace(style_text_head, "")
            string_center = string_center.replace(style_text_tail, "")
            string_center_list = string_center.split("\n")
        else:
            string_center_list = string_list
    for i in range(0, len(string_list)):
        width_i = int(width) / 2
        count_index = text_handle.get_text_index(string_center_list[i])
        count_i = int(count_index) / 2
        if cache_contorl.wframe_mouse.w_frame_re_print:
            normal_print("\n")
            normal_print(" " * int((width_i - count_i)))
            normal_print(string_list[i])
        else:
            normal_print(" " * int((width_i - count_i)))
            one_by_one_print(sleep_time, string_list[i])
            normal_print("\n")
            if cache_contorl.wframe_mouse.w_frame_lines_up:
                index_i_up = i + 1
                cache_contorl.wframe_mouse.w_frame_lines_up = 2
                for index_i_up in range(index_i_up, len(string_list)):
                    restart_line_print(
                        text_handle.align(string_list[index_i_up], "center"), style,
                    )
                cache_contorl.wframe_mouse.w_frame_lines_state = 2
                break
    cache_contorl.wframe_mouse.w_frame_re_print = 0


# 多行回车逐行输出
def multiple_line_return_print(string=""):
    """
    绘制多行文本，并在绘制时，当玩家输入回车，才绘制下一行
    Keyword arguments:
    string -- 要绘制的文本
    """
    text_wait = cache_contorl.text_wait
    if text_wait != 0:
        time.sleep(text_wait)
    cache_contorl.wframe_mouse.w_frame_mouse_next_line = 1
    string_list = string.split("\n")
    for i in range(0, len(string_list)):
        line_feed_wait_print(string_list[i])
    cache_contorl.wframe_mouse.w_frame_mouse_next_line = 0
