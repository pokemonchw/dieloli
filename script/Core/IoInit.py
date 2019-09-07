# -*- coding: UTF-8 -*-
import threading
import queue
import json
import sys
from script.Core import MainFrame,GameConfig

# 防止系统爆栈，设置较多的栈深度
sys.setrecursionlimit(100000)

input_evnet = threading.Event()
_send_queue = queue.Queue()
_order_queue = queue.Queue()
order_swap = None

def _input_evnet_set(order):
    '''
    推送一个命令
    Keyword arguments:
    order -- 命令
    '''
    putOrder(order)

def getorder():
    '''
    获取一个命令
    '''
    return _order_queue.get()

MainFrame.bind_return(_input_evnet_set)
MainFrame.bind_queue(_send_queue)

def _get_input_event():
    '''
    获取输入事件锁
    '''
    return input_evnet

def run(open_func):
    '''
    运行游戏
    Keyword arguments:
    open_func -- 开场流程函数
    '''
    global _flowthread
    _flowthread = threading.Thread(target=open_func, name='flowthread')
    _flowthread.start()
    MainFrame._run()

def putQ(message):
    '''
    向输出队列中推送信息
    Keyword arguments:
    message -- 推送的信息
    '''
    _send_queue.put_nowait(message)

def putOrder(message):
    '''
    向命令队列中推送信息
    Keyword arguments:
    message -- 推送的信息
    '''
    _order_queue.put_nowait(message)

# #######################################################################
# json 构建函数

def new_json():
    '''
    定义一个通用json结构
    '''
    flowjson = {}
    flowjson['content'] = []
    return flowjson

def text_json(string, style):
    '''
    定义一个文本json
    Keyword arguments:
    string -- 要显示的文本
    style -- 显示时的样式
    '''
    re = {}
    re['type'] = 'text'
    re['text'] = string
    if type(style) == tuple:
        re['style'] = style
    if type(style) == type(''):
        re['style'] = (style,)
    return re

def cmd_json(cmd_str, cmd_num, normal_style, on_style):
    '''
    定义一个命令json
    Keyword arguments:
    cmd_str -- 命令文本
    cmd_num -- 命令数字
    normal_style -- 正常显示样式
    on_style -- 鼠标在其上时显示样式
    '''
    re = {}
    re['type'] = 'cmd'
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
    '''
    定义一个样式json
    Keyword arguments:
    style_name -- 样式名称
    foreground -- 前景色/字体颜色
    background -- 背景色
    font -- 字体
    fontsize -- 字号
    bold -- 加粗
    underline -- 下划线
    italic -- 斜体
    '''
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

def eraPrint(string, style='standard'):
    '''
    输出命令
    Keyword arguments:
    string -- 输出文本
    style -- 显示样式
    '''
    jsonstr = new_json()
    jsonstr['content'].append(text_json(string, style))
    putQ(json.dumps(jsonstr, ensure_ascii=False))

def imageprint(imageName,imagePath=''):
    '''
    图片输出命令
    Keyword arguments:
    imageName -- 图片名称
    imagePath -- 图片路径
    '''
    jsonstr = new_json()
    imageJson = {"imageName":imageName,"imagePath":imagePath}
    jsonstr['image'] = imageJson
    putQ(json.dumps(jsonstr, ensure_ascii=False))

def clear_screen():
    '''
    清屏
    '''
    jsonstr = new_json()
    jsonstr['clear_cmd'] = 'true'
    putQ(json.dumps(jsonstr, ensure_ascii=False))

def frame_style_def(style_name, foreground, background, font, fontsize, bold, underline, italic):
    '''
    推送一条在前端定义样式的信息
    Keyword arguments:
    style_name -- 样式名称
    foreground -- 前景色/字体颜色
    background -- 背景色
    font -- 字体
    fontsize -- 字号
    bold -- 加粗， 用1表示使用
    underline -- 下划线，用1表示使用
    italic -- 斜体，用1表示使用
    '''
    jsonstr = new_json()
    jsonstr['set_style'] = style_json(style_name, foreground, background, font, fontsize, bold, underline, italic)
    putQ(json.dumps(jsonstr, ensure_ascii=False))

def set_background(color):
    '''
    设置前端背景颜色
    Keyword arguments:
    color -- 颜色
    '''
    jsonstr = new_json()
    jsonstr['bgcolor'] = color
    putQ(json.dumps(jsonstr, ensure_ascii=False))

def clearorder():
    '''
    清楚前端已经设置的命令
    '''
    jsonstr = new_json()
    jsonstr['clearorder_cmd'] = 'true'
    putQ(json.dumps(jsonstr, ensure_ascii=False))

# ############################################################

# 命令生成函数
def io_print_cmd(cmd_str, cmd_number, normal_style='standard', on_style='onbutton'):
    '''
    打印一条指令
    Keyword arguments:
    cmd_str -- 命令文本
    cmd_number -- 命令数字
    normal_style -- 正常显示样式
    on_style -- 鼠标在其上时显示样式
    '''
    jsonstr = new_json()
    jsonstr['content'].append(cmd_json(cmd_str, cmd_number, normal_style, on_style))
    putQ(json.dumps(jsonstr, ensure_ascii=False))

# 清除命令函数
def io_clear_cmd(*cmd_numbers):
    '''
    清除命令
    Keyword arguments:
    cmd_number -- 命令数字，不输入则清楚当前已有的全部命令
    '''
    jsonstr = new_json()
    if cmd_numbers:
        jsonstr['clearcmd_cmd'] = cmd_numbers
    else:
        jsonstr['clearcmd_cmd'] = 'all'
    putQ(json.dumps(jsonstr, ensure_ascii=False))

def style_def():
    pass

def init_style():
    '''
    富文本样式初始化
    '''
    global style_def
    def new_style_def(style_name, foreground, background, font, fontsize,bold, underline, italic):
        frame_style_def(style_name, foreground, background, font, fontsize, bold, underline, italic)
    style_def = new_style_def
    styleList = GameConfig.getFontDataList()
    standardData = GameConfig.getFontData('standard')
    styleDataList = ['foreground','background','font','fontSize','bold','underline','italic']
    defStyleList = {}
    for i in range(0,len(styleList)):
        styleName = styleList[i]
        styleData = GameConfig.getFontData(styleName)
        for index in range(0,len(styleDataList)):
            try:
                styleDataValue = styleData[styleDataList[index]]
            except KeyError:
                styleDataValue = standardData[styleDataList[index]]
            defStyleList[styleDataList[index]] = styleDataValue
        styleForeground = defStyleList['foreground']
        styleBackground = defStyleList['background']
        styleFont = defStyleList['font']
        styleFontSize = defStyleList['fontSize']
        styleBold = defStyleList['bold']
        styleUnderline = defStyleList['underline']
        styleItalic = defStyleList['italic']
        style_def(styleName, styleForeground, styleBackground, styleFont, styleFontSize, styleBold, styleUnderline,
                  styleItalic)
