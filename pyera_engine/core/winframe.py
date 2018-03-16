# -*- coding: UTF-8 -*-
import os
from tkinter import *
from tkinter import ttk
import json
import uuid
import core.GameConfig as config
import core.TextLoading as textload
import core.CacheContorl as cache
import core.SettingFrame as settingframe
import core.TextHandle as text
from tkinter import font

def closeWindow():
    os._exit(0)

# 显示主框架
gameName = config.game_name
root = Tk()
root.title(gameName)
root.geometry(config.window_width + 'x' + config.window_hight + '+0+0')
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
root.protocol('WM_DELETE_WINDOW', closeWindow)
mainframe = ttk.Frame(root,borderwidth = 2)
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)

# 显示窗口
textbox = Text(mainframe, width=config.textbox_width, height=config.textbox_hight,
               highlightbackground = config.background_color,bd = 0)
textbox.grid(column=0, row=0, sticky=(N, W, E, S))

# 垂直滚动条
s_vertical = ttk.Scrollbar(mainframe, orient=VERTICAL, command=textbox.yview)
textbox.configure(yscrollcommand=s_vertical.set)
s_vertical.grid(column=1, row=0, sticky=(N, E, S),rowspan=2)

# 输入框背景容器
orderFontData = textload.getTextData(textload.fontListId,'order')
inputBackgroundBox = Text(mainframe,highlightbackground = config.background_color,background = config.background_color,bd = 0)
inputBackgroundBox.grid(column=0, row=1, sticky=(W, E, S))

cursorText = config.cursor
cursorWidth = text.getTextIndex(cursorText)
inputBackgroundBoxCursor = Text(inputBackgroundBox,width=cursorWidth, height=1,highlightbackground = orderFontData['background'],background = orderFontData['background'],bd = 0)
inputBackgroundBoxCursor.grid(column=0, row=0, sticky=(W, E, S))
inputBackgroundBoxCursor.insert('end',cursorText)
inputBackgroundBoxCursor.config(foreground=orderFontData['foreground'])

# 输入栏
estyle = ttk.Style()
estyle.element_create("plain.field", "from", "clam")
estyle.layout(
    "EntryStyle.TEntry",
    [(
        'Entry.plain.field',{
            'children':[(
                'Entry.background',{
                    'children':[(
                        'Entry.padding',{
                            'children':[(
                                'Entry.textarea',{
                                    'sticky':'nswe'
                                }
                            )],
                            'sticky':'nswe'
                        }
                    )],
                    'sticky':'nswe'
                }
            )],
            'border':'0',
            'sticky':'nswe'
        }
    )]
)
estyle.configure("EntryStyle.TEntry",
                 background=orderFontData['background'],
                 foreground=orderFontData['foreground'],
                 selectbackground = orderFontData['selectbackground']
                 )
order = StringVar()
orderFont = font.Font(family = orderFontData['font'],size = orderFontData['fontSize'])
inputboxWidth = int(config.textbox_width) - cursorWidth
inputbox = ttk.Entry(inputBackgroundBox, style = 'EntryStyle.TEntry',textvariable=order,font = orderFont,width = inputboxWidth)
inputbox.grid(column=1, row=0, sticky=(N, E, S))


# 构建菜单栏
root.option_add('*tearOff', FALSE)
menubar = Menu(root)
root['menu'] = menubar
menufile = Menu(menubar)
menutest = Menu(menubar)
menuother = Menu(menubar)
menubar.add_cascade(menu=menufile, label=textload.getTextData(textload.menuId,textload.menuFile))
menubar.add_cascade(menu=menuother, label=textload.getTextData(textload.menuId,textload.menuOther))

def reset(*args):
    cache.flowContorl['restartGame'] = 1
    send_input()

def quit(*args):
    cache.flowContorl['quitGame'] = 1
    send_input()

def setting(*args):
    settingframe.openSettingFrame()

menufile.add_command(label=textload.getTextData(textload.menuId,textload.menuRestart),command=reset)
menufile.add_command(label=textload.getTextData(textload.menuId,textload.menuQuit),command=quit)

menuother.add_command(label=textload.getTextData(textload.menuId,textload.menuSetting),command=setting)
menuother.add_command(label=textload.getTextData(textload.menuId,textload.menuAbout))

input_event_func = None
send_order_state = False
# when false, send 'skip'; when true, send cmd

def send_input(*args):
    global input_event_func
    order = _getorder()
    if len(cache.inputCache) >= 21:
        if(order) == '':
            pass
        else:
            del cache.inputCache[0]
            cache.inputCache.append(order)
            cache.inputPosition['position'] = 0
    else:
        if (order) == '':
            pass
        else:
            cache.inputCache.append(order)
            cache.inputPosition['position'] = 0
    input_event_func(order)
    _clearorder()

# #######################################################################
# 运行函数
_flowthread = None

def read_queue():
    while not _queue.empty():
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
        if 'bgcolor' in jsonstr.keys() :
            set_background(jsonstr['bgcolor'])
        if 'set_style' in jsonstr.keys():
            temp=jsonstr['set_style']
            _frame_style_def(temp['style_name'],temp['foreground'],temp['background'],temp['font'],
                             temp['fontsize'],temp['bold'],temp['underline'],temp['italic'])
        if 'image' in jsonstr.keys():
            import core.EraImage as eraimage
            eraimage.printImage(jsonstr['image']['imageName'],jsonstr['image']['imagePath'])

        for c in jsonstr['content']:
            if c['type']=='text':
                _print(c['text'], style=tuple(c['style']))
            if c['type'] == 'cmd':
                _io_print_cmd(c['text'],c['num'])
    root.after(10, read_queue)

def _run():
    root.after(10, read_queue)
    root.mainloop()

def seeend():
    textbox.see(END)

def set_background(color):
    textbox.config(insertbackground=color)
    textbox.configure(background=color, selectbackground="red")

# ######################################################################
# ######################################################################
# ######################################################################
# 双框架公共函数

_queue = None

def bind_return(func):
    global input_event_func
    input_event_func = func

def bind_queue(q):
    global _queue
    _queue = q

# #######################################################################
# 输出格式化

sysprint = print

def _print(string,style=('standard',)):
    textbox.insert(END,string,style)
    seeend()

def _printCmd(string,style=('standard',)):
    textbox.insert('end', string, style)
    seeend()

def _clear_screen():
    _io_clear_cmd()
    textbox.delete('1.0', END)

def _frame_style_def(style_name, foreground, background, font, fontsize, bold, underline, italic):
    font_list = []
    font_list.append(font)
    font_list.append(fontsize)
    if bold == '1':
        font_list.append('bold')
    if underline == '1':
        font_list.append('underline')
    if italic == '1':
        font_list.append('italic')
    textbox.tag_configure(style_name, foreground=foreground, background=background, font=tuple(font_list))

# #########################################################3
# 输入处理函数

def _getorder():
    return order.get()

def setorder(orderstr):
    order.set(orderstr)

def _clearorder():
    order.set('')

# ############################################################

cmd_tag_map = {}

# 命令生成函数
def _io_print_cmd(cmd_str, cmd_number, normal_style='standard', on_style='onbutton'):
    global cmd_tag_map
    cmd_tagname = str(uuid.uuid1())
    textbox.tag_configure(cmd_tagname)
    if cmd_number in cmd_tag_map:
        _io_clear_cmd(cmd_number)
    cmd_tag_map[cmd_number] = cmd_tagname

    def send_cmd(*args):
        global send_order_state
        send_order_state=True
        order.set(cmd_number)
        send_input(order)

    def enter_func(*args):
        textbox.tag_remove(normal_style, textbox.tag_ranges(cmd_tagname)[0], textbox.tag_ranges(cmd_tagname)[1])
        textbox.tag_add(on_style, textbox.tag_ranges(cmd_tagname)[0], textbox.tag_ranges(cmd_tagname)[1])
        cache.wframeMouse['mouseLeaveCmd'] = 0

    def leave_func(*args):
        textbox.tag_add(normal_style, textbox.tag_ranges(cmd_tagname)[0], textbox.tag_ranges(cmd_tagname)[1])
        textbox.tag_remove(on_style, textbox.tag_ranges(cmd_tagname)[0], textbox.tag_ranges(cmd_tagname)[1])
        cache.wframeMouse['mouseLeaveCmd'] = 1

    textbox.tag_bind(cmd_tagname, '<1>', send_cmd)
    textbox.tag_bind(cmd_tagname, '<Enter>', enter_func)
    textbox.tag_bind(cmd_tagname, '<Leave>', leave_func)
    _printCmd(cmd_str, style=(cmd_tagname, normal_style))


# 清除命令函数
def _io_clear_cmd(*cmd_numbers):
    global cmd_tag_map
    if cmd_numbers:
        for num in cmd_numbers:
            if num in cmd_tag_map:
                index_first = textbox.tag_ranges(cmd_tag_map[num])[0]
                index_last = textbox.tag_ranges(cmd_tag_map[num])[1]
                for tag_name in textbox.tag_names(index_first):
                    textbox.tag_remove(tag_name, index_first, index_last)
                textbox.tag_add('standard', index_first, index_last)
                textbox.tag_delete(cmd_tag_map[num])
                cmd_tag_map.pop(num)
    else:
        for num in cmd_tag_map.keys():
            index_first = textbox.tag_ranges(cmd_tag_map[num])[0]
            index_lskip_one_waitast = textbox.tag_ranges(cmd_tag_map[num])[1]
            for tag_name in textbox.tag_names(index_first):
                textbox.tag_remove(tag_name, index_first, index_lskip_one_waitast)
            textbox.tag_add('standard', index_first, index_lskip_one_waitast)
            textbox.tag_delete(cmd_tag_map[num])
        cmd_tag_map.clear()