import os
import sys
import uuid
import psutil
import signal
import threading
import time
from queue import Queue, Empty

from PySide6.QtCore import (
    Qt, Signal, Slot,
    QTimer, QObject, QThread,
)
from PySide6.QtGui import (
    QColor, QFont, QFontMetrics,
    QTextCharFormat, QTextImageFormat, QImage,
    QCursor, QTextCursor, QPixmap,
    QTextDocument, QFontDatabase, QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QTextEdit, QLineEdit, QLabel,
    QVBoxLayout, QHBoxLayout
)

import screeninfo

from Script.Core import (
    text_handle, game_type, cache_control,
)
from Script.Config import normal_config, game_config

cache: game_type.Cache = cache_control.cache

class MainWindow(QMainWindow):
    """主窗口类"""

    def __init__(self):
        """初始化主窗口"""
        super().__init__()

        self.game_name = normal_config.config_normal.game_name
        self.setWindowTitle(self.game_name)

        # 初始化变量
        self.styles = {}
        self.input_event_func = None
        self.send_order_state = False
        self.main_queue = Queue()
        self.cmd_tag_map = {}
        self.user_scrolling = False  # 用户是否正在滚动的标志位
        self.current_hover_cmd = None  # 当前悬停的指令编号
        self.image_data = {}
        self.image_text_data = {}
        self.image_lock = 0

        # 设置窗口和字体
        self._setup_window_and_font()

        # 创建主部件和布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)  # 移除主布局的边距
        self.main_layout.setSpacing(0)  # 移除主布局的间距

        self.textbox_layout = QHBoxLayout()
        # 创建事件显示区域
        self.eventbox = CommandTextEdit(self)
        self.eventbox.setReadOnly(True)
        self.eventbox.setFont(self.normal_font)
        self.eventbox.setReadOnly(True)
        # 创建面板显示区域
        self.panelbox = CommandTextEdit(self)
        self.panelbox.setReadOnly(True)
        self.panelbox.setFont(self.normal_font)
        self.panelbox.setReadOnly(True)

        self.textbox_layout.addWidget(self.eventbox)
        self.textbox_layout.addWidget(self.panelbox)
        self.eventbox.hide()
        self.textbox_layout.setStretch(1, 1)
        self.textbox_layout.invalidate()
        self.main_layout.addLayout(self.textbox_layout)

        # 设置文本框背景颜色
        self.set_background(normal_config.config_normal.background)

        # 创建输入区域
        self._setup_input_area()

        # 初始化图片数据
        self.load_and_resize_images()

        # 启动定时器读取队列
        self.timer = QTimer()
        self.timer.timeout.connect(self.read_queue)
        self.timer.start(1)  # 每 1 毫秒检查一次队列

    def _setup_window_and_font(self):
        """设置窗口大小和字体"""
        # 获取屏幕信息
        screens = QApplication.screens()
        cursor_pos = QCursor.pos()
        current_screen = QApplication.screenAt(cursor_pos)
        screen_geometry = current_screen.geometry()
        screen_width = screen_geometry.width()
        screen_height = screen_geometry.height()

        # 计算窗口大小和字体大小
        window_width = screen_width - 45
        window_height = screen_height - 45
        need_char_width = window_width / normal_config.config_normal.textbox_width
        need_char_height = window_height / normal_config.config_normal.text_hight
        now_font_size = 20
        try:
            font_file_path = os.path.join("data", "font", "SarasaMonoSC-Regular.ttf")
            font_id = QFontDatabase.addApplicationFont(font_file_path)
            font_families = QFontDatabase.applicationFontFamilies(font_id)
            normal_config.config_normal.font = font_families[0]
            font_family = normal_config.config_normal.font
        except:
            pass

        font = QFont(font_family, now_font_size)
        font_metrics = QFontMetrics(font)
        char_width = font_metrics.horizontalAdvance('a')
        char_height = font_metrics.lineSpacing()

        if char_width <= need_char_width and char_height <= need_char_height:
            need_font_size = now_font_size
            while True:
                next_font_size = need_font_size + 1
                next_font = QFont(font_family, next_font_size)
                next_font_metrics = QFontMetrics(next_font)
                next_char_width = next_font_metrics.horizontalAdvance('a')
                next_char_height = next_font_metrics.lineSpacing()
                if next_char_width <= need_char_width and next_char_height <= need_char_height:
                    need_font_size = next_font_size
                else:
                    break
        else:
            need_font_size = now_font_size
            while True:
                next_font_size = need_font_size - 1
                if next_font_size <= 0:
                    break
                next_font = QFont(font_family, next_font_size)
                next_font_metrics = QFontMetrics(next_font)
                next_char_width = next_font_metrics.horizontalAdvance('a')
                next_char_height = next_font_metrics.lineSpacing()
                need_font_size = next_font_size
                if next_char_width <= need_char_width and next_char_height <= need_char_height:
                    break

        # 设置字体
        self.normal_font = QFont(font_family, need_font_size)
        normal_config.config_normal.font_size = need_font_size
        normal_config.config_normal.order_font_size = need_font_size

        font_metrics = QFontMetrics(self.normal_font)
        self.now_char_width = font_metrics.horizontalAdvance('a')
        self.now_char_height = font_metrics.lineSpacing()
        window_width = self.now_char_width * normal_config.config_normal.textbox_width
        window_height = self.now_char_height * normal_config.config_normal.text_hight

        # 设置窗口大小和位置
        win_width = window_width + 45  # 调整窗口框架的宽度
        win_height = window_height + 45  # 调整窗口框架的高度
        x = current_screen.geometry().x() + (current_screen.geometry().width() - win_width) // 2
        y = current_screen.geometry().y() + (current_screen.geometry().height() - win_height) // 2
        self.setGeometry(x, y, win_width, win_height)

    def _setup_input_area(self):
        """创建输入区域"""
        self.input_layout = QHBoxLayout()
        self.input_layout.setContentsMargins(0, 0, 0, 0)  # 移除输入布局的边距
        self.input_layout.setSpacing(0)  # 移除输入布局的间距

        self.prompt_label = QLabel("~$")
        self.prompt_label.setFont(self.normal_font)
        self.prompt_label.setStyleSheet(f"color: {game_config.config_font[0].foreground};")
        self.prompt_label.setContentsMargins(0, 0, 0, 0)  # 移除提示符的边距

        self.order = ""
        self.input_line = QLineEdit()
        self.input_line.setFont(self.normal_font)
        self.input_line.returnPressed.connect(self.send_input)
        self.input_line.setFocus()
        self.input_line.setFrame(False)  # 移除输入框的边框

        # 设置输入框背景颜色、字体颜色和移除边框
        self.input_line.setStyleSheet(f"""
            QLineEdit {{
                background-color: {game_config.config_font[0].background};
                color: {game_config.config_font[0].foreground};
                selection-background-color: {game_config.config_font[0].selectbackground};
                border: none;
                outline: none;
            }}
            QLineEdit:focus {{
                border: none;
                outline: none;
            }}
        """)

        # 创建一个容器部件来包含提示符和输入框，并设置背景颜色为黑色
        self.input_container = QWidget()
        self.input_container.setStyleSheet(f"background-color: {game_config.config_font[0].background};")
        self.input_container_layout = QHBoxLayout(self.input_container)
        self.input_container_layout.setContentsMargins(0, 0, 0, 0)
        self.input_container_layout.setSpacing(0)
        self.input_container_layout.addWidget(self.prompt_label)
        self.input_container_layout.addWidget(self.input_line)

        self.input_layout.addWidget(self.input_container)
        self.main_layout.addLayout(self.input_layout)

    def closeEvent(self, event):
        """关闭事件处理，关闭游戏进程"""
        self.close_window()
        event.accept()

    def close_window(self):
        """关闭游戏，会终止当前进程和所有子进程"""
        parent = psutil.Process(os.getpid())
        children = parent.children(recursive=True)
        for process in children:
            process.send_signal(signal.SIGTERM)
        os._exit(0)

    def send_input(self):
        """发送一条指令"""
        order_text = self.get_order()
        if len(cache.input_cache) >= 21:
            if order_text != "":
                del cache.input_cache[0]
                cache.input_cache.append(order_text)
                cache.input_position = 0
        else:
            if order_text != "":
                cache.input_cache.append(order_text)
                cache.input_position = 0
        if self.input_event_func:
            self.input_event_func(order_text)
        self.clear_order()

    def get_order(self):
        """获取命令框中的内容"""
        return self.input_line.text()

    def set_order(self, order_str):
        """设置命令框中的内容"""
        self.input_line.setText(order_str)

    def clear_order(self):
        """清空命令框"""
        self.input_line.clear()

    def bind_return(self, func):
        """
        绑定输入处理函数
        Keyword arguments：
        func -- 输入处理函数
        """
        self.input_event_func = func

    def bind_queue(self, q):
        """
        绑定信息队列
        Keyword arguments：
        q -- 消息队列
        """
        self.main_queue = q

    def open_eventbox(self):
        """打开事件文本面板"""
        self.eventbox.setFixedWidth(self.now_char_width*int(normal_config.config_normal.textbox_width/3))
        self.eventbox.show()
        self.textbox_layout.invalidate()

    @Slot()
    def read_queue(self):
        """从队列中获取在前端显示的信息"""
        while not self.main_queue.empty():
            json_data = self.main_queue.get()
            # 处理 json_data
            if 'clear_cmd' in json_data and json_data['clear_cmd'] == 'true':
                self.clear_screen()
            if 'clearorder_cmd' in json_data and json_data['clearorder_cmd'] == 'true':
                self.clear_order()
            if 'clearcmd_cmd' in json_data:
                cmd_nums = json_data['clearcmd_cmd']
                if cmd_nums == 'all':
                    self.io_clear_cmd()
                else:
                    self.io_clear_cmd(*cmd_nums)
            if 'bgcolor' in json_data:
                self.set_background(json_data['bgcolor'])
            if 'set_style' in json_data:
                temp = json_data['set_style']
                self.frame_style_def(
                    temp["style_name"],
                    temp["foreground"],
                    temp["background"],
                    temp["font"],
                    temp["fontsize"],
                    temp["bold"],
                    temp["underline"],
                    temp["italic"],
                )
            if 'image' in json_data:
                self.now_print_image(json_data['image']['image_name'])
            for c in json_data.get('content', []):
                if c["type"] == "text":
                    self.now_print(c["text"], style=tuple(c["style"]))
                if c["type"] == "cmd":
                    self.io_print_cmd(c["text"], c["num"], c["normal_style"][0], c["on_style"][0])
                if c["type"] == "image_cmd":
                    self.io_print_image_cmd(c["text"], c["num"])
                if c["type"] == "event":
                    self.event_print(c["text"], style=tuple(c["style"]))
                # 如果需要，实现滚动限制
        self.input_line.setFocus()

    def set_background(self, color: str):
        """
        设置背景颜色
        Keyword arguments:
        color -- 背景颜色字符串
        """
        pal = self.panelbox.palette()
        pal.setColor(self.panelbox.viewport().backgroundRole(), QColor(color))
        self.panelbox.setPalette(pal)
        self.panelbox.setStyleSheet(f"background-color: {color};")
        self.eventbox.setPalette(pal)
        self.eventbox.setStyleSheet(f"background-color: {color};")

    def now_print(self, string: str, style=('standard',)):
        """
        输出文本
        Keyword arguments:
        string -- 要输出的字符串
        style -- 样式名称的元组
        """
        cursor = self.panelbox.textCursor()
        text_format = QTextCharFormat()
        for style_name in style:
            if style_name in self.styles:
                text_format.merge(self.styles[style_name])
        cursor.insertText(string, text_format)
        self.panelbox.ensureCursorVisible()

    def event_print(self, string: str, style=('standard',)):
        """
        在事件面板绘制文本
        Keyword arguments:
        string -- 要输出的字符串
        style -- 样式名称的元组
        """
        self.eventbox.moveCursor(QTextCursor.End)
        cursor = self.eventbox.textCursor()
        text_format = QTextCharFormat()
        for style_name in style:
            if style_name in self.styles:
                text_format.merge(self.styles[style_name])
        cursor.insertText(string, text_format)
        self.eventbox.ensureCursorVisible()

    def frame_style_def(self, style_name: str, foreground: str, background: str, font_family: str, font_size: int, bold: int, underline: int, italic: int):
        """
        定义样式
        Keyword arguments:
        style_name -- 样式名称
        foreground -- 前景色（字体颜色）
        background -- 背景色
        font_family -- 字体家族
        font_size -- 字号
        bold -- 是否加粗（'1' 或 '0'）
        underline -- 是否下划线（'1' 或 '0'）
        italic -- 是否斜体（'1' 或 '0'）
        """
        text_format = QTextCharFormat()
        if foreground:
            text_format.setForeground(QColor(foreground))
        if background:
            text_format.setBackground(QColor(background))
        font = QFont(font_family, font_size)
        font.setBold(bold == '1')
        font.setUnderline(underline == '1')
        font.setItalic(italic == '1')
        text_format.setFont(font)
        self.styles[style_name] = text_format

    def io_print_cmd(self, cmd_str: str, cmd_return: str, normal_style='standard', on_style='onbutton'):
        """
        打印一条指令
        Keyword arguments:
        cmd_str -- 命令文本
        cmd_return -- 命令返回
        normal_style -- 正常显示样式
        on_style -- 鼠标悬停时的样式
        """
        self.panelbox.moveCursor(QTextCursor.End)
        cursor = self.panelbox.textCursor()
        start_pos = cursor.position()
        text_format = self.styles.get(normal_style, QTextCharFormat())
        cursor.insertText(cmd_str, text_format)
        end_pos = cursor.position()
        self.cmd_tag_map[cmd_return] = (start_pos, end_pos, normal_style, on_style)
        self.panelbox.ensureCursorVisible()

    @Slot(int)
    def send_cmd(self, cmd_return: str):
        """
        发送命令
        Keyword arguments:
        cmd_return -- 命令返回
        """
        self.send_order_state = True
        self.set_order(str(cmd_return))
        self.send_input()

    def io_clear_cmd(self, *cmd_returns):
        """
        清除命令
        Keyword arguments:
        cmd_returns: 要清除的命令返回，不传参数则清除所有命令
        """
        if cmd_returns:
            for cmd_return in cmd_returns:
                if cmd_return in self.cmd_tag_map:
                    del self.cmd_tag_map[cmd_number]
        else:
            self.cmd_tag_map.clear()

    def clear_screen(self):
        """清屏"""
        self.io_clear_cmd()
        self.panelbox.clear()

    def now_print_image(self, image_name: str):
        """
        输出图片
        Keyword arguments:
        image_name -- 图片名称（不含路径和扩展名）
        """
        self.panelbox.moveCursor(QTextCursor.End)
        cursor = self.panelbox.textCursor()
        image_format = QTextImageFormat()
        image = self.image_data[image_name]  # QImage 对象
        image_format.setName(image_name)  # 使用图片名称作为资源名称
        image_format.setWidth(image.width())
        image_format.setHeight(image.height())
        # 将图片作为资源添加到文档中
        self.panelbox.document().addResource(QTextDocument.ImageResource, image_name, image)
        cursor.insertImage(image_format)

    def io_print_image_cmd(self, image_path: str, cmd_return: str):
        """
        打印一个图片按钮
        Keyword arguments:
        image_path -- 图片路径
        cmd_return -- 点击图片响应的命令数字
        """
        self.panelbox.moveCursor(QTextCursor.End)
        cursor = self.panelbox.textCursor()
        start_pos = cursor.position()
        image_format = QTextImageFormat()
        image = self.image_data[image_name]  # QImage 对象
        image_format.setName(image_name)
        image_format.setWidth(image.width())
        image_format.setHeight(image.height())
        # 将图片作为资源添加到文档中
        self.panelbox.document().addResource(QTextDocument.ImageResource, image_name, image)
        cursor.insertImage(image_format)
        end_pos = cursor.position()
        self.cmd_tag_map[cmd_return] = (start_pos, end_pos, '', '')

    def load_and_resize_images(self):
        """加载并调整图片大小"""
        image_dir_path = os.path.join("image")
        if not os.path.exists(image_dir_path):
            print(f"图片目录 '{image_dir_path}' 不存在。")
            return
        for image_file_path_id in os.listdir(image_dir_path):
            image_file_path = os.path.join(image_dir_path, image_file_path_id)
            if not os.path.isfile(image_file_path):
                continue
            image_file_name = os.path.splitext(image_file_path_id)[0]  # 移除扩展名
            old_image = QImage(image_file_path)
            if old_image.isNull():
                print(f"无法加载图片 '{image_file_path}'。")
                continue
            old_width = old_image.width()
            old_height = old_image.height()
            font_width_scaling = self.now_char_width / 12
            font_height_scaling = self.now_char_height / 24
            now_width = int(old_width * font_width_scaling)
            now_height = int(old_height * font_height_scaling)
            # 调整图片大小
            new_image = old_image.scaled(now_width, now_height)
            # 存储图片
            self.image_data[image_file_name] = new_image  # 存储为 QImage 对象

    def eventFilter(self, obj, event):
        """事件过滤器，用于处理全局事件"""
        if event.type() == QEvent.MouseButtonPress:
            if event.button() == Qt.LeftButton:
                self.mouse_left_check(event)
            elif event.button() == Qt.RightButton:
                self.mouse_right_check(event)
        elif event.type() == QEvent.KeyPress:
            if event.key() == Qt.Key_Up:
                self.key_up(event)
            elif event.key() == Qt.Key_Down:
                self.key_down(event)
        return super().eventFilter(obj, event)

    def key_up(self, event):
        """键盘上键事件处理"""
        while cache.input_position == 0:
            cache.input_position = len(cache.input_cache)
        while 1 < cache.input_position <= 21:
            cache.input_position -= 1
            input_id = cache.input_position
            try:
                self.set_order(cache.input_cache[input_id])
                break
            except KeyError:
                cache.input_position += 1

    def key_down(self, event):
        """键盘下键事件处理"""
        if 0 <= cache.input_position < len(cache.input_cache) - 1:
            try:
                cache.input_position += 1
                input_id = cache.input_position
                self.set_order(cache.input_cache[input_id])
            except KeyError:
                cache.input_position -= 1
        elif cache.input_position == len(cache.input_cache) - 1:
            cache.input_position = 0
            self.set_order("")

    def mouse_check_push(self):
        """更正鼠标点击状态数据映射"""
        if cache.wframe_mouse.mouse_leave_cmd:
            self.send_input()
            cache.wframe_mouse.mouse_leave_cmd = 1

    def update_cmd_style(self, cmd_number, style_name):
        """更新指定命令的样式
        Keyword arguments:
        cmd_number -- 命令编号
        style_name -- 要应用的样式名称
        """
        if cmd_number in self.cmd_tag_map:
            start, end, normal_style, on_style = self.cmd_tag_map[cmd_number]
            cursor = self.panelbox.textCursor()
            cursor.setPosition(start)
            cursor.setPosition(end, QTextCursor.KeepAnchor)
            text_format = self.styles.get(style_name, QTextCharFormat())
            cursor.setCharFormat(text_format)

    def restore_previous_cmd_style(self):
        """恢复之前悬停的命令的样式"""
        if self.current_hover_cmd is not None:
            cmd_number = self.current_hover_cmd
            if cmd_number in self.cmd_tag_map:
                start, end, normal_style, on_style = self.cmd_tag_map[cmd_number]
                cursor = self.panelbox.textCursor()
                cursor.setPosition(start)
                cursor.setPosition(end, QTextCursor.KeepAnchor)
                text_format = self.styles.get(normal_style, QTextCharFormat())
                cursor.setCharFormat(text_format)
            self.current_hover_cmd = None

    def init_image_data(self):
        """ 初始化图像数据 """
        image_text_data = {}
        image_lock = 0
        image_dir_path = os.path.join("image")
        for image_file_path_id in os.listdir(image_dir_path):
            image_file_path = os.path.join(image_dir_path, image_file_path_id)
            image_file_name = image_file_path_id.rstrip(".png")
            old_image = QPixmap(image_file_path)
            old_height = old_image.height()
            old_width = old_image.width()
            font_width_scaling = main_frame.now_char_width / 12
            font_height_scaling = main_frame.now_char_height / 24
            new_height = int(old_height * font_height_scaling)
            new_width = int(old_width * font_width_scaling)
            new_image = old_image.scaled(new_width, new_height)
            self.image_data[image_file_name] = new_image



class CommandTextEdit(QTextEdit):
    """自定义的文本编辑器，用于处理命令点击事件和鼠标悬停事件"""

    commandClicked = Signal(int)

    def __init__(self, main_window):
        """初始化"""
        super().__init__()
        self.setMouseTracking(True)
        self.setFrameStyle(QTextEdit.NoFrame)  # 移除文本编辑器的边框
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)  # 设置垂直滚动条策略
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)  # 隐藏水平滚动条
        self.main_window = main_window  # 引用主窗口
        self.setCursor(Qt.IBeamCursor)  # 设置初始光标为文本光标
        self.setReadOnly(True)  # 设置为只读，禁用编辑

    def mousePressEvent(self, event):
        """鼠标点击事件处理"""
        if event.button() == Qt.LeftButton:
            cursor = self.cursorForPosition(event.pos())
            position = cursor.position()
            if self.main_window and hasattr(self.main_window, 'cmd_tag_map'):
                for cmd_number, (start, end, normal_style, on_style) in self.main_window.cmd_tag_map.items():
                    if start <= position <= end:
                        self.main_window.send_cmd(cmd_number)
                        return
            self.main_window.input_line.setFocus()
            self.main_window.mouse_check_push()
            return
        elif event.button() == Qt.RightButton:
            # 处理鼠标右键事件
            self.on_right_click(event)
            return  # 阻止事件进一步传播，防止出现默认菜单
        super().mousePressEvent(event)


    def on_right_click(self, event):
        """处理鼠标右键点击事件"""
        cache.wframe_mouse.mouse_right = 1
        cache.text_wait = 0
        cache.wframe_mouse.w_frame_skip_wait_mouse = 1
        self.main_window.mouse_check_push()
        event.accept()

    def contextMenuEvent(self, event):
        """覆盖上下文菜单事件，防止默认菜单出现"""
        # 不调用父类方法，直接返回，禁用右键菜单
        pass

    def mouseMoveEvent(self, event):
        """鼠标移动事件处理"""
        cursor = self.cursorForPosition(event.pos())
        position = cursor.position()
        main_window = self.main_window
        found = False
        if main_window and hasattr(main_window, 'cmd_tag_map'):
            for cmd_number, (start, end, normal_style, on_style) in main_window.cmd_tag_map.items():
                if start <= position <= end:
                    self.viewport().setCursor(Qt.PointingHandCursor)
                    if main_window.current_hover_cmd != cmd_number:
                        # 恢复之前的样式
                        main_window.restore_previous_cmd_style()
                        # 更新当前的悬停指令编号
                        main_window.current_hover_cmd = cmd_number
                        # 更新当前指令的样式为 on_style
                        main_window.update_cmd_style(cmd_number, on_style)
                    found = True
                    break
        if not found:
            self.viewport().setCursor(Qt.IBeamCursor)
            # 恢复之前的样式
            main_window.restore_previous_cmd_style()
        super().mouseMoveEvent(event)


app = QApplication(sys.argv)
window = MainWindow()

def run():
    window.show()
    app.exec()
