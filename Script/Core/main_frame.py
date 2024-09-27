# -*- coding: UTF-8 -*-
import os
import math
import uuid
import psutil
import signal
from types import FunctionType
from tkinter import (
    ttk,
    Tk,
    Text,
    StringVar,
    END,
    N,
    W,
    E,
    S,
    VERTICAL,
    font,
    Entry,
)
import pyautogui
import screeninfo
from Script.Core import (
    text_handle,
    game_type,
    cache_control,
)
from Script.Config import normal_config, game_config

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """


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
game_name = normal_config.config_normal.game_name
root = Tk()
now_font_size = 20
# 获取所有屏幕的信息
screens = screeninfo.get_monitors()
current_screen = None
x, y = pyautogui.position()
for screen in screens:
    if screen.x <= x <= screen.x + screen.width and screen.y <= y <= screen.y + screen.height:
        current_screen = screen
screen_width = current_screen.width
screen_height = current_screen.height
window_width = screen_width - 30
window_height = screen_height - 30
need_char_width = int(window_width / 120)
need_char_height = int(window_height / 50)
test_font = font.Font(family=normal_config.config_normal.font,size=now_font_size)
char_width = test_font.measure("a")
char_height = test_font.metrics("linespace")
if char_width <= need_char_width and char_height <= need_char_height:
    need_font_size = now_font_size
    while True:
        next_font_size = need_font_size + 1
        next_font = font.Font(family=normal_config.config_normal.font,size=next_font_size)
        next_font_char_width = next_font.measure("a")
        next_font_char_height = next_font.metrics("linespace")
        if next_font_char_width <= need_char_width and next_font_char_height <= need_char_height:
            need_font_size = next_font_size
        else:
            break
else:
    need_font_size = now_font_size
    while True:
        next_font_size = need_font_size - 1
        next_font = font.Font(family=normal_config.config_normal.font,size=next_font_size)
        next_font_char_width = next_font.measure("a")
        next_font_char_height = next_font.metrics("linespace")
        need_font_size = next_font_size
        if next_font_char_width <= need_char_width and next_font_char_height <= need_char_height:
            break
if normal_config.config_normal.font_size:
    need_font_size = normal_config.config_normal.font_size
now_font = font.Font(family=normal_config.config_normal.font,size=need_font_size)
now_char_width = now_font.measure("a")
now_char_height = now_font.metrics("linespace")
window_width = now_char_width * 120
window_height = now_char_height * 50
normal_config.config_normal.font_size = need_font_size
normal_config.config_normal.order_font_size = need_font_size
root.title(game_name)
width = window_width + 30
frm_width = root.winfo_rootx() - root.winfo_x()
win_width = width + 2 * frm_width
height = window_height
titlebar_height = root.winfo_rooty() - root.winfo_y()
win_height = height + titlebar_height + frm_width
x = current_screen.x + (current_screen.width // 2 - win_width // 2)
y = current_screen.y + (current_screen.height // 2 - win_height // 2)
root.geometry("{}x{}+{}+{}".format(win_width, win_height, x, y))
root.deiconify()
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
root.protocol("WM_DELETE_WINDOW", close_window)
main_frame = ttk.Frame(root, borderwidth=2)
main_frame.grid(column=0, row=0, sticky=(N, W, E, S))
main_frame.columnconfigure(0, weight=1)
main_frame.rowconfigure(0, weight=1)

normal_font = font.Font(family=normal_config.config_normal.font, size=need_font_size)
# 显示窗口
textbox = Text(
    main_frame,
    width=normal_config.config_normal.textbox_width,
    height=normal_config.config_normal.textbox_hight,
    highlightbackground=normal_config.config_normal.background,
    bd=0,
    cursor="",
    font=normal_font
)
textbox.grid(column=0, row=0, sticky=(N, W, E, S))

# 垂直滚动条
s_vertical = ttk.Scrollbar(main_frame, orient=VERTICAL, command=textbox.yview)
textbox.configure(yscrollcommand=s_vertical.set)
s_vertical.grid(column=1, row=0, sticky=(N, E, S), rowspan=2)

# 输入框背景容器
order_font_data = game_config.config_font[1]
for k in game_config.config_font[0].__dict__:
    if k not in order_font_data.__dict__:
        order_font_data.__dict__[k] = game_config.config_font[0].__dict__[k]
        order_font_data.font = normal_config.config_normal.font
        order_font_data.font_size = normal_config.config_normal.order_font_size
input_background_box = Text(
    main_frame,
    highlightbackground=normal_config.config_normal.background,
    background=normal_config.config_normal.background,
    bd=0,
)
input_background_box.grid(column=0, row=1, sticky=(W, E, S))

order_font = font.Font(family=order_font_data.font, size=order_font_data.font_size)
cursor_text = "~$"
cursor_width = text_handle.get_text_index(cursor_text)
input_background_box_cursor = Text(
    input_background_box,
    width=cursor_width,
    height=1,
    highlightbackground=order_font_data.background,
    background=order_font_data.background,
    bd=0,
    font=order_font,
)
input_background_box_cursor.grid(column=0, row=0, sticky=(W, E, S))
input_background_box_cursor.insert("end", cursor_text)
input_background_box_cursor.config(foreground=order_font_data.foreground)

# 输入栏
order = StringVar()
inputbox = Entry(
    input_background_box,
    borderwidth=0,
    insertwidth=1,
    insertbackground=order_font_data.foreground,
    selectborderwidth=0,
    highlightthickness=0,
    bg=order_font_data.background,
    foreground=order_font_data.foreground,
    selectbackground=order_font_data.selectbackground,
    textvariable=order,
    font=order_font,
    width=normal_config.config_normal.inputbox_width,
)
inputbox.grid(column=1, row=0, sticky=(N, E, S))
input_event_func = None
send_order_state = False
# when false, send 'skip'; when true, send cmd
def get_widget_font(widget):
    # 获取小部件的字体属性
    widget_font = font.nametofont(widget.cget("font"))
    # 返回字体的实际名称
    return widget_font.actual()

from Script.Core import era_image

def send_input(*args):
    """
    发送一条指令
    """
    global input_event_func
    order = get_order()
    if len(cache.input_cache) >= 21:
        if (order) != "":
            del cache.input_cache[0]
            cache.input_cache.append(order)
            cache.input_position = 0
    else:
        if (order) != "":
            cache.input_cache.append(order)
            cache.input_position = 0
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
        json_data = main_queue.get()

        if "clear_cmd" in json_data and json_data["clear_cmd"] == "true":
            clear_screen()
        if "clearorder_cmd" in json_data and json_data["clearorder_cmd"] == "true":
            clear_order()
        if "clearcmd_cmd" in json_data:
            cmd_nums = json_data["clearcmd_cmd"]
            if cmd_nums == "all":
                io_clear_cmd()
            else:
                io_clear_cmd(tuple(cmd_nums))
        if "bgcolor" in json_data:
            set_background(json_data["bgcolor"])
        if "set_style" in json_data:
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
        if "image" in json_data:
            textbox.image_create(
                "end", image=era_image.image_data[json_data["image"]["image_name"]]
            )

        for c in json_data["content"]:
            if c["type"] == "text":
                now_print(c["text"], style=tuple(c["style"]))
            if c["type"] == "cmd":
                io_print_cmd(c["text"], c["num"], c["normal_style"], c["on_style"])
            if c["type"] == "image_cmd":
                io_print_image_cmd(c["text"], c["num"])
            if (
                    "\n" in c["text"]
                    and textbox.get("1.0", END).count("\n")
                    > normal_config.config_normal.text_hight * 10
            ):
                textbox.delete("1.0", str(normal_config.config_normal.text_hight * 5) + ".0")
    root.after(1, read_queue)


def run():
    """
    启动屏幕
    """
    root.after(1, read_queue)
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


cmd_tag_map = {}


def io_print_cmd(cmd_str: str, cmd_number: int, normal_style="standard", on_style="onbutton"):
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
        """发送命令"""
        global send_order_state
        send_order_state = True
        order.set(cmd_number)
        textbox.configure(cursor="")
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
        cache.wframe_mouse.mouse_leave_cmd = 0

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
        textbox.configure(cursor="")
        cache.wframe_mouse.mouse_leave_cmd = 1

    textbox.tag_bind(cmd_tag_name, "<1>", send_cmd)
    textbox.tag_bind(cmd_tag_name, "<Enter>", enter_func)
    textbox.tag_bind(cmd_tag_name, "<Leave>", leave_func)
    print_cmd(cmd_str, style=(cmd_tag_name, normal_style))


def io_print_image_cmd(cmd_str: str, cmd_number: int):
    """
    打印一个图片按钮
    Keyword arguments:
    cmd_str -- 图片id
    cmd_number -- 点击图片响应数字
    """
    global cmd_tag_map
    cmd_tag_name = str(uuid.uuid1())
    textbox.tag_configure(cmd_tag_name)
    if cmd_number in cmd_tag_map:
        io_clear_cmd(cmd_number)
    cmd_tag_map[cmd_number] = cmd_tag_name

    def send_cmd(*args):
        """发送命令"""
        global send_order_state
        send_order_state = True
        order.set(cmd_number)
        textbox.configure(cursor="")
        send_input(order)

    index = textbox.index("end -1c")
    textbox.image_create(index, image=era_image.image_data[cmd_str])
    textbox.tag_add(cmd_tag_name, index, "{0} + 1 char".format(index))
    textbox.tag_bind(cmd_tag_name, "<1>", send_cmd)
    see_end()


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
        for num in cmd_tag_map:
            tag_tuple = textbox.tag_ranges(cmd_tag_map[num])
            if tag_tuple:
                index_first = tag_tuple[0]
                index_lskip_one_waitast = tag_tuple[1]
                for tag_name in textbox.tag_names(index_first):
                    textbox.tag_remove(tag_name, index_first, index_lskip_one_waitast)
                textbox.tag_add("standard", index_first, index_lskip_one_waitast)
            textbox.tag_delete(cmd_tag_map[num])
        cmd_tag_map.clear()
