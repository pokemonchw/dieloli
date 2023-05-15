# -*- coding: UTF-8 -*-
import os
import queue
import threading
import uuid
import psutil
import signal
from typing import Dict
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTextEdit, QLineEdit
from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor, QTextCharFormat, QFont, QColor
from Script.Core import (
    text_handle,
    game_type,
    cache_control,
)
from Script.Config import normal_config, game_config

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
font_map: Dict[str, QTextCharFormat] = {}
""" 所有字体的对象数据 """
input_event_func = None
send_order_state = False
main_queue = queue.Queue()
order_queue = queue.Queue()
window = None

app = QApplication([])


class AddParameterStr(str):
    """ 增加了UI控制参数的字符串对象 """

    def __init__(self, now_str: str):
        super().__init__()
        self.default_style: str = ""
        """ 默认的状态 """
        self.on_mouse_style: str = ""
        """ 鼠标悬停时的状态 """
        self.bind_order: str = ""
        """ 被点击后触发的指令 """
        self.now_str = now_str

    def __str__(self):
        return self.now_str


def bind_return(func):
    """
    绑定输入处理函数
    Keyword arguments:
    func -- 输入处理函数
    """
    global input_event_func
    input_event_func = func


class MainApp(QWidget):
    def __init__(self):
        super().__init__()
        width = normal_config.config_normal.window_width
        height = normal_config.config_normal.window_hight
        self.resize(width, height)
        self.setWindowTitle(normal_config.config_normal.game_name)
        self.order = ""
        self.order_line = QLineEdit(self)
        self.main_text = QTextEdit(self)
        self.main_text.setReadOnly(True)
        self.init_font_data()
        layout = QVBoxLayout()
        layout.addWidget(self.main_text)
        layout.addWidget(self.order_line)
        self.setLayout(layout)
        self.order_line.returnPressed.connect(self.on_return)
        self.main_text.cursorPositionChanged.connect(self.handle_curror_position_changed)
        self.current_word: AddParameterStr = AddParameterStr("")
        """ 鼠标当前选中的字符 """
        self.read_thread = threading.Thread(target=self.read_queue)
        self.read_thread.start()

    def read_queue(self):
        """ 读取消息队列 """
        while 1:
            now_data = main_queue.get()
            print(now_data)

    def init_font_data(self):
        """ 初始化所有字体样式对象 """
        normal_font_config = None
        for font_id in game_config.config_font:
            font_config = game_config.config_font[font_id]
            if normal_font_config == None:
                normal_font_config = font_config
            for now_key in normal_font_config.__dict__:
                if now_key not in font_config.__dict__:
                    font_config.__dict__[now_key] = normal_font_config.__dict__[now_key]
            now_style = QTextCharFormat()
            now_font = QFont()
            now_font.setFamily(normal_config.config_normal.font)
            now_font.setBold(font_config.bold)
            now_font.setItalic(font_config.italic)
            now_style.setFont(now_font)
            foreground = QColor(font_config.foreground)
            background = QColor(font_config.background)
            now_style.setForeground(foreground)
            now_style.setBackground(background)
            now_style.setFontPointSize(normal_config.config_normal.font_size)
            font_map[font_config.name] = now_style

    def handle_curror_position_changed(self):
        """ 处理当前光标位置的文本 """
        text_cursor = self.main_text.textCursor()
        text_cursor.select(QTextCursor.WordUnderCursor)
        if self.current_word.default_style != "":
            text_cursor.mergeCharFormat(font_map[self.current_word.default_style])
        else:
            text_cursor.mergeCharFormat(font_map["standard"])
        word: AddParameterStr = text_cursor.selectedText()
        # 更改样式或触发其他效果
        if word:
            if word.on_mouse_style:
                text_cursor.select(word)
                text_cursor.mergeCharFormat(font_map[word.on_mouse_style])
                self.current_word = word

    def on_return(self):
        """
        发送一条指令
        """
        global input_event_func
        order = self.get_order()
        if len(cache.input_cache) >= 21:
            if (order) != "":
                del cache.input_cache[0]
                cache.input_cache.append(order)
                cache.input_position = 0
        else:
            if (order) != "":
                cache.input_cache.append(order)
                cache.input_position = 0
        order_queue.put_nowait(order)
        self.clear_order()

    def get_order(self):
        """
        获取命令框中的内容
        """
        return self.order_line.text()

    def set_order(self, order_str: str):
        """
        设置命令框中内容
        """
        self.order_line.setText(order_str)

    def clear_order(self):
        """
        清空命令框
        """
        self.order_line.clear()

    def now_print(self, string, style=("standard",)):
        """
        输出文本
        Keyword arguments:
        string -- 字符串
        style -- 样式序列
        """
        self.main_text.append(string)  # TODO: Add support for styles

    def io_print_cmd(self, cmd_str: str, cmd_number: int, normal_style="standard", on_style="onbutton"):
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
        if cmd_number in cmd_tag_map:
            pass
        cursor = self.main_text.textCursor()
        cursor.movePosition(QTextCursor.END)
        now_position = cursor.position()
        now_style = font_map[normal_style]
        cursor.setCharFormat(now_style)
        cursor.insertText(cmd_str)

    def io_clear_cmd(self, *cmd_numbers: list):
        """
        清除命令
        Keyword arguments:
        cmd_number -- 命令数字，不输入则清楚当前已有的全部命令
        """
        global cmd_tag_map
        cursor: QTextCursor = self.main_text.textCursor()
        if cmd_numbers:
            for num in cmd_numbers:
                if num in cmd_tag_map:
                    cursor.movePosition(cmd_tag_map[num])
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


def clear_screen(self):
    """
    清屏
    """
    self.main_text.clear()


def close_window():
    """
    关闭游戏，会终止当前进程和所有子进程
    """
    parent = psutil.Process(os.getpid())
    children = parent.children(recursive=True)
    for process in children:
        process.send_signal(signal.SIGTERM)
    os._exit(0)


def start_main_window():
    """ 开始窗体线程 """
    global window
    window = MainApp()
    window.show()
    app.aboutToQuit.connect(close_window)
    app.exec_()
