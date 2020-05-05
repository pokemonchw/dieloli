# -*- coding: UTF-8 -*-
import os
import json
import uuid
import psutil
import signal
from tkinter import (
    ttk,
    Tk,
    Text,
    StringVar,
    FALSE,
    Menu,
    END,
    N,
    W,
    E,
    S,
    VERTICAL,
    font,
    Entry,
)
from Script.Core import (
    game_config,
    text_loading,
    cache_contorl,
    setting_frame,
    about_frame,
    text_handle,
    constant,
)


def close_window():
    """
    关闭游戏，会终止当前进程和所有子进程
    """
    parent = psutil.Process(os.getpid())
    children = parent.children(recursive=True)
    for process in children:
        process.send_signal(signal.SIGTERM)
    os._exit(0)


# 显示主框架
game_name = game_config.game_name
root = Tk()
root.title(game_name)
root.geometry(
    game_config.window_width + "x" + game_config.window_hight + "+0+0"
)
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
root.protocol("WM_DELETE_WINDOW", close_window)
main_frame = ttk.Frame(root, borderwidth=2)
main_frame.grid(column=0, row=0, sticky=(N, W, E, S))
main_frame.columnconfigure(0, weight=1)
main_frame.rowconfigure(0, weight=1)

# 显示窗口
textbox = Text(
    main_frame,
    width=game_config.textbox_width,
    height=game_config.textbox_hight,
    highlightbackground=game_config.background_color,
    bd=0,
)
textbox.grid(column=0, row=0, sticky=(N, W, E, S))

# 垂直滚动条
s_vertical = ttk.Scrollbar(main_frame, orient=VERTICAL, command=textbox.yview)
textbox.configure(yscrollcommand=s_vertical.set)
s_vertical.grid(column=1, row=0, sticky=(N, E, S), rowspan=2)

# 输入框背景容器
order_font_data = text_loading.get_text_data(
    constant.FilePath.FONT_CONFIG_PATH, "order"
)
input_background_box = Text(
    main_frame,
    highlightbackground=game_config.background_color,
    background=game_config.background_color,
    bd=0,
)
input_background_box.grid(column=0, row=1, sticky=(W, E, S))

cursor_text = game_config.cursor
cursor_width = text_handle.get_text_index(cursor_text)
input_background_box_cursor = Text(
    input_background_box,
    width=cursor_width,
    height=1,
    highlightbackground=order_font_data["background"],
    background=order_font_data["background"],
    bd=0,
)
input_background_box_cursor.grid(column=0, row=0, sticky=(W, E, S))
input_background_box_cursor.insert("end", cursor_text)
input_background_box_cursor.config(foreground=order_font_data["foreground"])

# 输入栏
order = StringVar()
order_font = font.Font(
    family=order_font_data["font"], size=order_font_data["fontSize"]
)
inputbox = Entry(
    input_background_box,
    borderwidth=0,
    insertborderwidth=0,
    selectborderwidth=0,
    highlightthickness=0,
    bg=order_font_data["background"],
    foreground=order_font_data["foreground"],
    selectbackground=order_font_data["selectbackground"],
    textvariable=order,
    font=order_font,
    width=game_config.inputbox_width,
)
inputbox.grid(column=1, row=0, sticky=(N, E, S))

# 构建菜单栏
root.option_add("*tear_off", FALSE)
menu_bar = Menu(root)
root["menu"] = menu_bar
menu_file = Menu(menu_bar)
menu_test = Menu(menu_bar)
menu_other = Menu(menu_bar)
menu_bar.add_cascade(
    menu=menu_file,
    label=text_loading.get_text_data(
        constant.FilePath.MENU_PATH, constant.WindowMenu.MENU_FILE
    ),
)
menu_bar.add_cascade(
    menu=menu_other,
    label=text_loading.get_text_data(
        constant.FilePath.MENU_PATH, constant.WindowMenu.MENU_OTHER
    ),
)


def reset(*args):
    """
    重置游戏
    """
    cache_contorl.flow_contorl["restart_game"] = 1
    send_input()


def quit(*args):
    """
    退出游戏
    """
    cache_contorl.flow_contorl["quit_game"] = 1
    send_input()


def setting(*args):
    """
    打开设置面板
    """
    setting_frame.open_setting_frame()


def about(*args):
    """
    打开关于面板
    """
    about_frame.open_about_frame()


menu_file.add_command(
    label=text_loading.get_text_data(
        constant.FilePath.MENU_PATH, constant.WindowMenu.MENU_RESTART
    ),
    command=reset,
)
menu_file.add_command(
    label=text_loading.get_text_data(
        constant.FilePath.MENU_PATH, constant.WindowMenu.MENU_QUIT
    ),
    command=quit,
)

menu_other.add_command(
    label=text_loading.get_text_data(
        constant.FilePath.MENU_PATH, constant.WindowMenu.MENU_SETTING
    ),
    command=setting,
)
menu_other.add_command(
    label=text_loading.get_text_data(
        constant.FilePath.MENU_PATH, constant.WindowMenu.MENU_ABBOUT
    ),
    command=about,
)

input_event_func = None
send_order_state = False
# when false, send 'skip'; when true, send cmd


def send_input(*args):
    """
    发送一条指令
    """
    global input_event_func
    order = get_order()
    if len(cache_contorl.input_cache) >= 21:
        if not (order) == "":
            del cache_contorl.input_cache[0]
            cache_contorl.input_cache.append(order)
            cache_contorl.input_position["position"] = 0
    else:
        if not (order) == "":
            cache_contorl.input_cache.append(order)
            cache_contorl.input_position["position"] = 0
    input_event_func(order)
    clear_order()


# #######################################################################
# 运行函数
flow_thread = None


def read_queue():
    """
    从队列中获取在前端显示的信息
    """
    while not main_queue.empty():
        quene_str = main_queue.get()
        json_data = json.loads(quene_str)

        if (
            "clear_cmd" in json_data.keys()
            and json_data["clear_cmd"] == "true"
        ):
            clear_screen()
        if (
            "clearorder_cmd" in json_data.keys()
            and json_data["clearorder_cmd"] == "true"
        ):
            clear_order()
        if "clearcmd_cmd" in json_data.keys():
            cmd_nums = json_data["clearcmd_cmd"]
            if cmd_nums == "all":
                io_clear_cmd()
            else:
                io_clear_cmd(tuple(cmd_nums))
        if "bgcolor" in json_data.keys():
            set_background(json_data["bgcolor"])
        if "set_style" in json_data.keys():
            temp = json_data["set_style"]
            frame_style_def(
                temp["style_name"],
                temp["foreground"],
                temp["background"],
                temp["font"],
                temp["fontsize"],
                temp["bold"],
                temp["underline"],
                temp["italic"],
            )
        if "image" in json_data.keys():
            from Script.Core import era_image

            era_image.print_image(
                json_data["image"]["image_name"],
                json_data["image"]["image_path"],
            )

        for c in json_data["content"]:
            if c["type"] == "text":
                now_print(c["text"], style=tuple(c["style"]))
            if c["type"] == "cmd":
                io_print_cmd(
                    c["text"], c["num"], c["normal_style"], c["on_style"]
                )
    root.after(10, read_queue)


def run():
    """
    启动屏幕
    """
    root.after(10, read_queue)
    root.mainloop()


def see_end():
    """
    输出END信息
    """
    textbox.see(END)


def set_background(color):
    """
    设置背景颜色
    Keyword arguments:
    color -- 背景颜色
    """
    textbox.config(insertbackground=color)
    textbox.configure(background=color, selectbackground="red")


# ######################################################################
# ######################################################################
# ######################################################################
# 双框架公共函数

main_queue = None


def bind_return(func):
    """
    绑定输入处理函数
    Keyword arguments:
    func -- 输入处理函数
    """
    global input_event_func
    input_event_func = func


def bind_queue(q):
    """
    绑定信息队列
    Keyword arguments:
    q -- 消息队列
    """
    global main_queue
    main_queue = q


# #######################################################################
# 输出格式化

sys_print = print


def now_print(string, style=("standard",)):
    """
    输出文本
    Keyword arguments:
    string -- 字符串
    style -- 样式序列
    """
    textbox.insert(END, string, style)
    see_end()


def print_cmd(string, style=("standard",)):
    """
    输出文本
    Keyword arguments:
    string -- 字符串
    style -- 样式序列
    """
    textbox.insert("end", string, style)
    see_end()


def clear_screen():
    """
    清屏
    """
    io_clear_cmd()
    textbox.delete("1.0", END)


def frame_style_def(
    style_name: str,
    foreground: str,
    background: str,
    font: str,
    font_size: int,
    bold: str,
    under_line: str,
    italic: str,
):
    """
    定义样式
    Keyword arguments:
    style_name -- 样式名称
    foreground -- 前景色/字体颜色
    background -- 背景色
    font -- 字体
    fontsize -- 字号
    bold -- 加粗
    underline -- 下划线
    italic -- 斜体
    """
    font_list = []
    font_list.append(font)
    font_list.append(font_size)
    if bold == "1":
        font_list.append("bold")
    if under_line == "1":
        font_list.append("underline")
    if italic == "1":
        font_list.append("italic")
    textbox.tag_configure(
        style_name,
        foreground=foreground,
        background=background,
        font=tuple(font_list),
    )


# #########################################################3
# 输入处理函数


def get_order():
    """
    获取命令框中的内容
    """
    return order.get()


def set_order(order_str: str):
    """
    设置命令框中内容
    """
    order.set(order_str)


def clear_order():
    """
    清空命令框
    """
    order.set("")


# ############################################################

cmd_tag_map = {}


# 命令生成函数
def io_print_cmd(
    cmd_str: str, cmd_number: int, normal_style="standard", on_style="onbutton"
):
    """
    打印一条指令
    Keyword arguments:
    cmd_str -- 命令文本
    cmd_number -- 命令数字
    normal_style -- 正常显示样式
    on_style -- 鼠标在其上时显示样式
    """
    global cmd_tag_map
    cmd_tag_name = str(uuid.uuid1())
    textbox.tag_configure(cmd_tag_name)
    if cmd_number in cmd_tag_map:
        io_clear_cmd(cmd_number)
    cmd_tag_map[cmd_number] = cmd_tag_name

    def send_cmd(*args):
        """
        发送命令
        """
        global send_order_state
        send_order_state = True
        order.set(cmd_number)
        send_input(order)

    def enter_func(*args):
        """
        鼠标进入改变命令样式
        """
        textbox.tag_remove(
            normal_style,
            textbox.tag_ranges(cmd_tag_name)[0],
            textbox.tag_ranges(cmd_tag_name)[1],
        )
        textbox.tag_add(
            on_style,
            textbox.tag_ranges(cmd_tag_name)[0],
            textbox.tag_ranges(cmd_tag_name)[1],
        )
        cache_contorl.wframe_mouse["mouse_leave_cmd"] = 0

    def leave_func(*args):
        """
        鼠标离开还原命令样式
        """
        textbox.tag_add(
            normal_style,
            textbox.tag_ranges(cmd_tag_name)[0],
            textbox.tag_ranges(cmd_tag_name)[1],
        )
        textbox.tag_remove(
            on_style,
            textbox.tag_ranges(cmd_tag_name)[0],
            textbox.tag_ranges(cmd_tag_name)[1],
        )
        cache_contorl.wframe_mouse["mouse_leave_cmd"] = 1

    textbox.tag_bind(cmd_tag_name, "<1>", send_cmd)
    textbox.tag_bind(cmd_tag_name, "<Enter>", enter_func)
    textbox.tag_bind(cmd_tag_name, "<Leave>", leave_func)
    print_cmd(cmd_str, style=(cmd_tag_name, normal_style))


# 清除命令函数
def io_clear_cmd(*cmd_numbers: list):
    """
    清除命令
    Keyword arguments:
    cmd_number -- 命令数字，不输入则清楚当前已有的全部命令
    """
    global cmd_tag_map
    if cmd_numbers:
        for num in cmd_numbers:
            if num in cmd_tag_map:
                index_first = textbox.tag_ranges(cmd_tag_map[num])[0]
                index_last = textbox.tag_ranges(cmd_tag_map[num])[1]
                for tag_name in textbox.tag_names(index_first):
                    textbox.tag_remove(tag_name, index_first, index_last)
                textbox.tag_add("standard", index_first, index_last)
                textbox.tag_delete(cmd_tag_map[num])
                del cmd_tag_map[num]
    else:
        for num in cmd_tag_map.keys():
            index_first = textbox.tag_ranges(cmd_tag_map[num])[0]
            index_lskip_one_waitast = textbox.tag_ranges(cmd_tag_map[num])[1]
            for tag_name in textbox.tag_names(index_first):
                textbox.tag_remove(
                    tag_name, index_first, index_lskip_one_waitast
                )
            textbox.tag_add("standard", index_first, index_lskip_one_waitast)
            textbox.tag_delete(cmd_tag_map[num])
        cmd_tag_map.clear()