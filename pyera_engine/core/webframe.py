# -*- coding: UTF-8 -*-


import json

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit, disconnect

binding_return_func = None
flowjson = {}
flowjson['content'] = []
order = 0
input_event_func = None

app = Flask(__name__)
app.debug = True
app.config['SECRET_KEY'] = ''
socketio = SocketIO(app, async_mode='eventlet')

flowthread = None

open_func = None
sysprint = print

# 启动
def _run():
    # socketio.run(app, host='0.0.0.0')
    socketio.run(app)

@app.route('/')
def interactive():
    return render_template('index.html')


@socketio.on('run', namespace='/test')
def test_connect(*args):
    global gamebegin_flag
    global flowthread
    global open_func

    try:
        if flowthread == None:
            flowthread = socketio.start_background_task(target=read_queue)

    except Exception as e:
        return str(e)


@socketio.on('dealorder', namespace='/test')
def test_message(value):
    sysprint('dealorder')
    try:
        setorder(value)
        send_input()
    except Exception as e:
        return str(e)


@socketio.on('connect', namespace='/test')
def test_connect():
    sysprint('connected')
    setorder('_reset_this_game_')
    send_input()

# #######################################################################
# json 构建函数

def _init_flowjson():
    global flowjson
    flowjson.clear()
    flowjson['content'] = []


def _new_json():
    flowjson = {}
    flowjson['content'] = []
    return flowjson


def _text_json(string, style):
    re = {}
    re['type'] = 'text'
    re['text'] = string.replace('\n', '<br/>')
    re['style'] = style
    return re


def _cmd_json(cmd_str, cmd_num, normal_style, on_style):
    re = {}
    re['type'] = 'cmd'
    re['text'] = cmd_str.replace('\n', '<br/>')
    re['num'] = cmd_num
    re['normal_style'] = normal_style
    re['on_style'] = on_style
    return re


def _style_json(style_name, foreground, background, font, fontsize, bold, underline, italic):
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
# 运行逻辑

def send_input(*args):
    global input_event_func
    order = getorder()
    input_event_func(order)
    _clearorder()


def set_background(color):
    jsonstr = _new_json()
    jsonstr['bgcolor'] = color
    socketio.emit('game_display', json.dumps(jsonstr, ensure_ascii=False), namespace='/test')


# ######################################################################
# ######################################################################
sumup=0
sumtime=0
def read_queue():
    while True:
        if not _queue.empty():
            quenestr = _queue.get()
            jsonstr = json.loads(quenestr)

            if 'clear_cmd' in jsonstr.keys() and jsonstr['clear_cmd'] == 'true':
                _clear_screen()
            if 'clearorder_cmd' in jsonstr.keys() and jsonstr['clearorder_cmd'] == 'true':
                _clearorder()
            if 'clearcmd_cmd' in jsonstr.keys():
                cmd_nums = jsonstr['clearcmd_cmd']
                if cmd_nums == "all":
                    _io_clear_cmd()
                else:
                    _io_clear_cmd(tuple(cmd_nums))
            if 'bgcolor' in jsonstr.keys():
                set_background(jsonstr['bgcolor'])
            if 'set_style' in jsonstr.keys():
                temp = jsonstr['set_style']
                _frame_style_def(temp['style_name'], temp['foreground'], temp['background'], temp['font'],
                                 temp['fontsize'], temp['bold'], temp['underline'], temp['italic'])
            for c in jsonstr['content']:
                if c['type'] == 'text':
                    _print(c['text'], style=' '.join(c['style']))
                if c['type'] == 'cmd':
                    _io_print_cmd(c['text'], c['num'], normal_style=' '.join(c['normal_style']), on_style=' '.join(c['on_style']))
                    # _io_print_cmd(c['text'], c['num'])
        socketio.sleep(0.01)
        global sumup, sumtime
        sumup +=1
        if sumup >200:
            sumup=0
            sumtime +=1
            sysprint(sumtime)

# 双框架公共函数
_queue = None

def bind_return(func):
    global input_event_func
    input_event_func = func
    return

def bind_queue(q):
    global _queue
    _queue = q

# #######################################################################
# 输出格式化

# @copy_current_request_context
def _print(string, style='standard'):
    jsonstr = _new_json()
    jsonstr['content'].append(_text_json(string, style))
    socketio.emit('game_display', json.dumps(jsonstr, ensure_ascii=False), namespace='/test')


def _clear_screen():
    # io_clear_cmd()
    jsonstr = _new_json()
    jsonstr['clear_cmd'] = 'true'
    socketio.emit('game_display', json.dumps(jsonstr, ensure_ascii=False), namespace='/test')


def _frame_style_def(style_name, foreground, background, font, fontsize, bold, underline, italic):
    jsonstr = _new_json()
    jsonstr['set_style'] = _style_json(style_name, foreground, background, font, fontsize, bold, underline, italic)
    socketio.emit('game_display', json.dumps(jsonstr, ensure_ascii=False), namespace='/test')


# #########################################################3
# 输入处理函数

def getorder():
    global order
    return order


def setorder(orderstr):
    global order
    order = orderstr


def _clearorder():
    jsonstr = _new_json()
    jsonstr['clearorder_cmd'] = 'true'
    socketio.emit('game_display', json.dumps(jsonstr, ensure_ascii=False), namespace='/test')


# ############################################################

# 命令生成函数
def _io_print_cmd(cmd_str, cmd_number, normal_style='standard', on_style='onbutton'):
    jsonstr = _new_json()
    jsonstr['content'].append(_cmd_json(cmd_str, cmd_number, normal_style, on_style))
    socketio.emit('game_display', json.dumps(jsonstr, ensure_ascii=False), namespace='/test')


# 清除命令函数
def _io_clear_cmd(*cmd_numbers):
    jsonstr = _new_json()
    if cmd_numbers:
        jsonstr['clearcmd_cmd'] = cmd_numbers
    else:
        jsonstr['clearcmd_cmd'] = 'all'
    socketio.emit('game_display', json.dumps(jsonstr, ensure_ascii=False), namespace='/test')


####################################################################
####################################################################



