# -*- coding: UTF-8 -*-
import os
import core.pycfg
import core.winframe as winframe
import threading
import queue
import json
import sys
import core.data as data
from core.pycfg import gamepath
sys_print = print

sys.setrecursionlimit(100000)

input_evnet = threading.Event()
_send_queue = queue.Queue()
_order_queue = queue.Queue()
order_swap = None


def _input_evnet_set(order):
    putOrder(order)


def getorder():
    return _order_queue.get()


winframe.bind_return(_input_evnet_set)
winframe.bind_queue(_send_queue)


def _get_input_event():
    return input_evnet

def run(open_func):
    global _flowthread
    _flowthread = threading.Thread(target=open_func, name='flowthread')
    _flowthread.start()
    core.winframe._run()

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


def style_def():
    pass


def init_style():
    global style_def

    def new_style_def(style_name, foreground, background, font, fontsize,bold, underline, italic):
        frame_style_def(style_name, foreground, background, font, fontsize, bold, underline, italic)

    style_def = new_style_def
    fontConfigPath = os.path.join(gamepath, 'data', 'FontConfig.json')
    fontData = data._loadjson(fontConfigPath)
    styleList = fontData["styleList"]
    for i in range(0,len(styleList)):
        styleName = styleList[i]
        styleData = fontData[styleName]
        styleForeground = styleData['foreground']
        styleBackground = styleData['background']
        styleFont = styleData['font']
        styleFontSize = styleData['fontSize']
        if styleData['bold'] == '0':
            styleBold = False
        else:
            styleBold = True
        if styleData['underline'] == '0':
            styleUnderline = False
        else:
            styleUnderline = True
        if styleData['italic'] == '0':
            styleItalic = False
        else:
            styleItalic = True
        style_def(styleName,styleForeground,styleBackground,styleFont,styleFontSize,styleBold,styleUnderline,styleItalic)
