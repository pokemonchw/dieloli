from tkinter import Event
from Script.Core import main_frame, py_cmd, cache_contorl

wframe = main_frame.root


def on_wframe_listion():
    """
    对按键事件进行绑定
    """
    wframe.bind("<ButtonPress-1>", mouse_left_check)
    wframe.bind("<ButtonPress-3>", mouse_right_check)
    wframe.bind("<Return>", main_frame.send_input)
    wframe.bind("<KP_Enter>", main_frame.send_input)
    wframe.bind("<Up>", key_up)
    wframe.bind("<Down>", key_down)


def mouse_left_check(event: Event):
    """
    鼠标左键事件处理
    Keyword arguments:
    event -- 鼠标事件
    """
    if cache_contorl.wframe_mouse["w_frame_up"] == 0:
        set_wframe_up()
    else:
        mouse_check_push()


def mouse_right_check(event: Event):
    """
    鼠标右键事件处理
    Keyword arguments:
    event -- 鼠标事件
    """
    cache_contorl.wframe_mouse["mouse_right"] = 1
    cache_contorl.text_wait = 0
    if cache_contorl.wframe_mouse["w_frame_up"] == 0:
        set_wframe_up()
    else:
        mouse_check_push()


def key_up(event: Event):
    """
    键盘上键事件处理
    Keyword arguments:
    event -- 键盘事件
    """
    while cache_contorl.input_position["position"] == 0:
        cache_contorl.input_position["position"] = len(cache_contorl.input_cache)
    while (
        cache_contorl.input_position["position"] <= 21
        and cache_contorl.input_position["position"] > 1
    ):
        cache_contorl.input_position["position"] = (
            cache_contorl.input_position["position"] - 1
        )
        inpot_id = cache_contorl.input_position["position"]
        try:
            main_frame.order.set(cache_contorl.input_cache[inpot_id])
            break
        except KeyError:
            cache_contorl.input_position["position"] = (
                cache_contorl.input_position["position"] + 1
            )


def key_down(event: Event):
    """
    键盘下键事件处理
    Keyword arguments:
    event -- 键盘事件
    """
    if (
        cache_contorl.input_position["position"] > 0
        and cache_contorl.input_position["position"]
        < len(cache_contorl.input_cache) - 1
    ):
        try:
            cache_contorl.input_position["position"] = (
                cache_contorl.input_position["position"] + 1
            )
            input_id = cache_contorl.input_position["position"]
            main_frame.order.set(cache_contorl.input_cache[input_id])
        except KeyError:
            cache_contorl.input_position["position"] = (
                cache_contorl.input_position["position"] - 1
            )
    elif cache_contorl.input_position["position"] == len(cache_contorl.input_cache) - 1:
        cache_contorl.input_position["position"] = 0
        main_frame.order.set("")


def set_wframe_up():
    """
    修正逐字输出状态为nowait
    """
    cache_contorl.wframe_mouse["w_frame_up"] = 1
    cache_contorl.wframe_mouse["w_frame_lines_up"] = 1


def mouse_check_push():
    """
    更正鼠标点击状态数据映射
    """
    py_cmd.focus_cmd()
    if cache_contorl.wframe_mouse["mouse_leave_cmd"] == 0:
        main_frame.send_input()
        cache_contorl.wframe_mouse["mouse_leave_cmd"] = 1
