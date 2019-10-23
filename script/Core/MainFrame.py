# -*- coding: UTF-8 -*-
import os
import json
import uuid
import psutil
import signal
from tkinter import ttk,Tk,Text,StringVar,FALSE,Menu,END,N,W,E,S,VERTICAL,font
from script.Core import GameConfig,TextLoading,CacheContorl,SettingFrame,AboutFrame,TextHandle


def closeWindow():
    '''
    关闭游戏，会终止当前进程和所有子进程
    '''
    parent = psutil.Process(os.getpid())
    children = parent.children(recursive=True)
    for process in children:
        process.send_signal(signal.SIGTERM)
    os._exit(0)

# 显示主框架
gameName = GameConfig.game_name
root = Tk()

def check_font():
	font_dict = {a:0 for a in font.families()}
	if 'Inconsolata' not in font_dict: #判断用户的操作系统类型
		if sys.platform == 'win32': 
			import ctypes
			from ctypes import wintypes

			try:
				import winreg
			except ImportError:
				import _winreg as winreg
			user32 = ctypes.WinDLL('user32', use_last_error=True)
			gdi32 = ctypes.WinDLL('gdi32', use_last_error=True)

			FONTS_REG_PATH = r'Software\Microsoft\Windows NT\CurrentVersion\Fonts'

			HWND_BROADCAST   = 0xFFFF
			SMTO_ABORTIFHUNG = 0x0002
			WM_FONTCHANGE    = 0x001D
			GFRI_DESCRIPTION = 1
			GFRI_ISTRUETYPE  = 3

			if not hasattr(wintypes, 'LPDWORD'):
				wintypes.LPDWORD = ctypes.POINTER(wintypes.DWORD)

			user32.SendMessageTimeoutW.restype = wintypes.LPVOID
			user32.SendMessageTimeoutW.argtypes = (
				wintypes.HWND,   # hWnd
				wintypes.UINT,   # Msg
				wintypes.LPVOID, # wParam
				wintypes.LPVOID, # lParam
				wintypes.UINT,   # fuFlags
				wintypes.UINT,   # uTimeout
				wintypes.LPVOID) # lpdwResult

			gdi32.AddFontResourceW.argtypes = (
				wintypes.LPCWSTR,) # lpszFilename

			# http://www.undocprint.org/winspool/getfontresourceinfo
			gdi32.GetFontResourceInfoW.argtypes = (
				wintypes.LPCWSTR, # lpszFilename
				wintypes.LPDWORD, # cbBuffer
				wintypes.LPVOID,  # lpBuffer
				wintypes.DWORD)   # dwQueryType
				
				
			def install_font(src_path): #srcpath是字体路径
				# copy the font to the Windows Fonts folder
				dst_path = os.path.join(os.environ['SystemRoot'], 'Fonts',
										os.path.basename(src_path))
				shutil.copy(src_path, dst_path)
				# load the font in the current session
				if not gdi32.AddFontResourceW(dst_path):
					os.remove(dst_path)
					raise WindowsError('AddFontResource failed to load "%s"' % src_path)
				# notify running programs
				user32.SendMessageTimeoutW(HWND_BROADCAST, WM_FONTCHANGE, 0, 0,
										   SMTO_ABORTIFHUNG, 1000, None)
				# store the fontname/filename in the registry
				filename = os.path.basename(dst_path)
				fontname = os.path.splitext(filename)[0]
				# try to get the font's real name
				cb = wintypes.DWORD()
				if gdi32.GetFontResourceInfoW(filename, ctypes.byref(cb), None,
											  GFRI_DESCRIPTION):
					buf = (ctypes.c_wchar * cb.value)()
					if gdi32.GetFontResourceInfoW(filename, ctypes.byref(cb), buf,
												  GFRI_DESCRIPTION):
						fontname = buf.value
				is_truetype = wintypes.BOOL()
				cb.value = ctypes.sizeof(is_truetype)
				gdi32.GetFontResourceInfoW(filename, ctypes.byref(cb),
					ctypes.byref(is_truetype), GFRI_ISTRUETYPE)
				if is_truetype:
					fontname += ' (TrueType)'
				with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, FONTS_REG_PATH, 0,
									winreg.KEY_SET_VALUE) as key:
					winreg.SetValueEx(key, fontname, 0, winreg.REG_SZ, filename)
			#安装字体
			install_font(r'.\..\..\data\font\Inconsolata-Bold.ttf')
			install_font(r'.\..\..\data\font\Inconsolata-Regular.ttf')
			return True #安装完之后返回True
			
		else:
			from shutil import copyfile #为文件复制做准备
			if not os.path.isdir(os.path.join(os.path.expandvars('$HOME'), 'Font'): #用户目录下没有Font就新建一个
				font_path = os.path.join(os.path.expandvars('$HOME'), 'Font')
				os.mkdir(font_path)
				#复制文件到/home/Font下
				copyfile(r'./../../data/font/Inconsolata-Bold.ttf',font_path + r'/Inconsolata-Bold.ttf')
				copyfile(r'./../../data/font/Inconsolata-Regular.ttf',font_path + r'/Inconsolata-Regular.ttf')
				return True #安装完之后返回True

#运行函数检查字体,True就不重建，否则重建一个Tk,这部分没有写。。
check_font()
	
root.title(gameName)
root.geometry(GameConfig.window_width + 'x' + GameConfig.window_hight + '+0+0')
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)
root.protocol('WM_DELETE_WINDOW', closeWindow)
mainframe = ttk.Frame(root,borderwidth = 2)
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
mainframe.columnconfigure(0, weight=1)
mainframe.rowconfigure(0, weight=1)

# 显示窗口
textbox = Text(mainframe, width=GameConfig.textbox_width, height=GameConfig.textbox_hight,
               highlightbackground = GameConfig.background_color,bd = 0)
textbox.grid(column=0, row=0, sticky=(N, W, E, S))

# 垂直滚动条
s_vertical = ttk.Scrollbar(mainframe, orient=VERTICAL, command=textbox.yview)
textbox.configure(yscrollcommand=s_vertical.set)
s_vertical.grid(column=1, row=0, sticky=(N, E, S),rowspan=2)

# 输入框背景容器
orderFontData = TextLoading.getTextData(TextLoading.fontConfigPath,'order')
inputBackgroundBox = Text(mainframe,highlightbackground = GameConfig.background_color,background = GameConfig.background_color,bd = 0)
inputBackgroundBox.grid(column=0, row=1, sticky=(W, E, S))

cursorText = GameConfig.cursor
cursorWidth = TextHandle.getTextIndex(cursorText)
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
inputboxWidth = int(GameConfig.textbox_width) - cursorWidth
inputbox = ttk.Entry(inputBackgroundBox, style = 'EntryStyle.TEntry',textvariable=order,font = orderFont,width = inputboxWidth)
inputbox.grid(column=1, row=0, sticky=(N, E, S))


# 构建菜单栏
root.option_add('*tearOff', FALSE)
menubar = Menu(root)
root['menu'] = menubar
menufile = Menu(menubar)
menutest = Menu(menubar)
menuother = Menu(menubar)
menubar.add_cascade(menu=menufile, label=TextLoading.getTextData(TextLoading.menuPath,TextLoading.menuFile))
menubar.add_cascade(menu=menuother, label=TextLoading.getTextData(TextLoading.menuPath,TextLoading.menuOther))

def reset(*args):
    '''
    重置游戏
    '''
    CacheContorl.flowContorl['restartGame'] = 1
    send_input()

def quit(*args):
    '''
    退出游戏
    '''
    CacheContorl.flowContorl['quitGame'] = 1
    send_input()

def setting(*args):
    '''
    打开设置面板
    '''
    SettingFrame.openSettingFrame()

def about(*args):
    '''
    打开关于面板
    '''
    AboutFrame.openAboutFrame()

menufile.add_command(label=TextLoading.getTextData(TextLoading.menuPath,TextLoading.menuRestart),command=reset)
menufile.add_command(label=TextLoading.getTextData(TextLoading.menuPath,TextLoading.menuQuit),command=quit)

menuother.add_command(label=TextLoading.getTextData(TextLoading.menuPath,TextLoading.menuSetting),command=setting)
menuother.add_command(label=TextLoading.getTextData(TextLoading.menuPath,TextLoading.menuAbout),command=about)

input_event_func = None
send_order_state = False
# when false, send 'skip'; when true, send cmd

def send_input(*args):
    '''
    发送一条指令
    '''
    global input_event_func
    order = _getorder()
    if len(CacheContorl.inputCache) >= 21:
        if not (order) == '':
            del CacheContorl.inputCache[0]
            CacheContorl.inputCache.append(order)
            CacheContorl.inputPosition['position'] = 0
    else:
        if not (order) == '':
            CacheContorl.inputCache.append(order)
            CacheContorl.inputPosition['position'] = 0
    input_event_func(order)
    _clearorder()

# #######################################################################
# 运行函数
_flowthread = None

def read_queue():
    '''
    从队列中获取在前端显示的信息
    '''
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
            from script.Core import EraImage
            EraImage.printImage(jsonstr['image']['imageName'],jsonstr['image']['imagePath'])

        for c in jsonstr['content']:
            if c['type']=='text':
                _print(c['text'], style=tuple(c['style']))
            if c['type'] == 'cmd':
                _io_print_cmd(c['text'],c['num'])
    root.after(10, read_queue)

def _run():
    '''
    启动屏幕
    '''
    root.after(10, read_queue)
    root.mainloop()

def seeend():
    '''
    输出END信息
    '''
    textbox.see(END)

def set_background(color):
    '''
    设置背景颜色
    Keyword arguments:
    color -- 背景颜色
    '''
    textbox.config(insertbackground=color)
    textbox.configure(background=color, selectbackground="red")

# ######################################################################
# ######################################################################
# ######################################################################
# 双框架公共函数

_queue = None

def bind_return(func):
    '''
    绑定输入处理函数
    Keyword arguments:
    func -- 输入处理函数
    '''
    global input_event_func
    input_event_func = func

def bind_queue(q):
    '''
    绑定信息队列
    Keyword arguments:
    q -- 消息队列
    '''
    global _queue
    _queue = q

# #######################################################################
# 输出格式化

sysprint = print

def _print(string,style=('standard',)):
    '''
    输出文本
    Keyword arguments:
    string -- 字符串
    style -- 样式序列
    '''
    textbox.insert(END,string,style)
    seeend()

def _printCmd(string,style=('standard',)):
    '''
    输出文本
    Keyword arguments:
    string -- 字符串
    style -- 样式序列
    '''
    textbox.insert('end', string, style)
    seeend()

def _clear_screen():
    '''
    清屏
    '''
    _io_clear_cmd()
    textbox.delete('1.0', END)

def _frame_style_def(style_name, foreground, background, font, fontsize, bold, underline, italic):
    '''
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
    '''
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
    '''
    获取命令框中的内容
    '''
    return order.get()

def setorder(orderstr):
    '''
    设置命令框中内容
    '''
    order.set(orderstr)

def _clearorder():
    '''
    清空命令框
    '''
    order.set('')

# ############################################################

cmd_tag_map = {}

# 命令生成函数
def _io_print_cmd(cmd_str, cmd_number, normal_style='standard', on_style='onbutton'):
    '''
    打印一条指令
    Keyword arguments:
    cmd_str -- 命令文本
    cmd_number -- 命令数字
    normal_style -- 正常显示样式
    on_style -- 鼠标在其上时显示样式
    '''
    global cmd_tag_map
    cmd_tagname = str(uuid.uuid1())
    textbox.tag_configure(cmd_tagname)
    if cmd_number in cmd_tag_map:
        _io_clear_cmd(cmd_number)
    cmd_tag_map[cmd_number] = cmd_tagname

    def send_cmd(*args):
        '''
        发送命令
        '''
        global send_order_state
        send_order_state=True
        order.set(cmd_number)
        send_input(order)

    def enter_func(*args):
        '''
        鼠标进入改变命令样式
        '''
        textbox.tag_remove(normal_style, textbox.tag_ranges(cmd_tagname)[0], textbox.tag_ranges(cmd_tagname)[1])
        textbox.tag_add(on_style, textbox.tag_ranges(cmd_tagname)[0], textbox.tag_ranges(cmd_tagname)[1])
        CacheContorl.wframeMouse['mouseLeaveCmd'] = 0

    def leave_func(*args):
        '''
        鼠标离开还原命令样式
        '''
        textbox.tag_add(normal_style, textbox.tag_ranges(cmd_tagname)[0], textbox.tag_ranges(cmd_tagname)[1])
        textbox.tag_remove(on_style, textbox.tag_ranges(cmd_tagname)[0], textbox.tag_ranges(cmd_tagname)[1])
        CacheContorl.wframeMouse['mouseLeaveCmd'] = 1

    textbox.tag_bind(cmd_tagname, '<1>', send_cmd)
    textbox.tag_bind(cmd_tagname, '<Enter>', enter_func)
    textbox.tag_bind(cmd_tagname, '<Leave>', leave_func)
    _printCmd(cmd_str, style=(cmd_tagname, normal_style))


# 清除命令函数
def _io_clear_cmd(*cmd_numbers):
    '''
    清除命令
    Keyword arguments:
    cmd_number -- 命令数字，不输入则清楚当前已有的全部命令
    '''
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
                del cmd_tag_map[num]
    else:
        for num in cmd_tag_map.keys():
            index_first = textbox.tag_ranges(cmd_tag_map[num])[0]
            index_lskip_one_waitast = textbox.tag_ranges(cmd_tag_map[num])[1]
            for tag_name in textbox.tag_names(index_first):
                textbox.tag_remove(tag_name, index_first, index_lskip_one_waitast)
            textbox.tag_add('standard', index_first, index_lskip_one_waitast)
            textbox.tag_delete(cmd_tag_map[num])
        cmd_tag_map.clear()
