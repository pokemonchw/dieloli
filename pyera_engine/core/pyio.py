# -*- coding: UTF-8 -*-

import core.pycfg

sys_print = print

if core.pycfg.platform == 'web':
    str = 'from core.webframe import *'
    exec(str)

if core.pycfg.platform == 'win':
    from core.winframe import *

import threading
import queue
import json

import sys

sys.setrecursionlimit(100000)

input_evnet = threading.Event()
_send_queue = queue.Queue()
_order_queue = queue.Queue()
order_swap = None


def _input_evnet_set(order):
    putOrder(order)


def getorder():
    return _order_queue.get()


bind_return(_input_evnet_set)
bind_queue(_send_queue)


def _get_input_event():
    return input_evnet


# style设置
_foreground = '#C8C8C8'
_background = '#2C4A69'
_font = '微软雅黑'
_fontsize = '14'


def run(open_func):
    global _flowthread
    _flowthread = threading.Thread(target=open_func, name='flowthread')
    _flowthread.start()
    if core.pycfg.platform == 'web':
        core.webframe._run()
    if core.pycfg.platform == 'win':
        core.winframe._run()
    _order_queue.put_nowait('_exit_game_')


def putQ(message):
    _send_queue.put_nowait(message)

def putOrder(message):
    _order_queue.put_nowait(message)

# #######################################################################
# json 构建函数

def new_json():
    flowjson = {}
    flowjson['content'] = []
    return flowjson


def text_json(string, style):
    re = {}
    re['type'] = 'text'
    # re['text'] = string.replace('\n', '<br/>')
    re['text'] = string
    if type(style) == tuple:
        re['style'] = style
    if type(style) == type(''):
        re['style'] = (style,)
    return re


def cmd_json(cmd_str, cmd_num, normal_style, on_style):
    re = {}
    re['type'] = 'cmd'
    # re['text'] = cmd_str.replace('\n', '<br/>')
    re['text'] = cmd_str
    re['num'] = cmd_num
    if type(normal_style) == tuple:
        re['normal_style'] = normal_style
    if type(normal_style) == type(''):
        re['normal_style'] = (normal_style,)

    if type(on_style) == tuple:
        re['on_style'] = on_style
    if type(on_style) == type(''):
        re['on_style'] = (on_style,)
    return re


def style_json(style_name, foreground, background, font, fontsize, bold, underline, italic):
    re = {}
    re['style_name'] = style_name
    re['foreground'] = foreground
    re['background'] = background
    re['font'] = font
    re['fontsize'] = fontsize
    re['bold'] = bold
    re['underline'] = underline
    re['italic'] = italic
    return re


# #######################################################################
# 输出格式化

def print(string, style='standard'):
    jsonstr = new_json()
    jsonstr['content'].append(text_json(string, style))
    putQ(json.dumps(jsonstr, ensure_ascii=False))


def clear_screen():
    jsonstr = new_json()
    jsonstr['clear_cmd'] = 'true'
    putQ(json.dumps(jsonstr, ensure_ascii=False))


def frame_style_def(style_name, foreground, background, font, fontsize, bold, underline, italic):
    jsonstr = new_json()
    jsonstr['set_style'] = style_json(style_name, foreground, background, font, fontsize, bold, underline, italic)
    putQ(json.dumps(jsonstr, ensure_ascii=False))


def set_background(color):
    jsonstr = new_json()
    jsonstr['bgcolor'] = color
    putQ(json.dumps(jsonstr, ensure_ascii=False))

def clearorder():
    jsonstr = new_json()
    jsonstr['clearorder_cmd'] = 'true'
    putQ(json.dumps(jsonstr, ensure_ascii=False))

# ############################################################

# 命令生成函数
def io_print_cmd(cmd_str, cmd_number, normal_style='standard', on_style='onbutton'):
    jsonstr = new_json()
    jsonstr['content'].append(cmd_json(cmd_str, cmd_number, normal_style, on_style))
    putQ(json.dumps(jsonstr, ensure_ascii=False))


# 清除命令函数
def io_clear_cmd(*cmd_numbers):
    jsonstr = new_json()
    if cmd_numbers:
        jsonstr['clearcmd_cmd'] = cmd_numbers
    else:
        jsonstr['clearcmd_cmd'] = 'all'
    putQ(json.dumps(jsonstr, ensure_ascii=False))


def style_def(style_name, foreground=_foreground, background=_background, font=_font, fontsize=_fontsize, bold=False,
              underline=False, italic=False):
    pass


def init_style(foreground_c, background_c, onbutton_c, font, font_size):
    global style_def

    def new_style_def(style_name, foreground=foreground_c, background=background_c, font=font, fontsize=font_size,
                      bold=False, underline=False, italic=False):
        frame_style_def(style_name, foreground, background, font, fontsize, bold, underline, italic)

    style_def = new_style_def
    style_def('standard')
    style_def('onbutton', foreground=onbutton_c)
