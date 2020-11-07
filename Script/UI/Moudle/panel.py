from typing import List, Dict
from types import FunctionType
from Script.UI.Moudle import draw
from Script.Core import io_init, flow_handle, text_handle, get_text,value_handle
from Script.Config import normal_config


_: FunctionType = get_text._
""" 翻译api """


class SingleColumnButton:
    """ 标准单列按钮监听响应 """

    def __init__(self):
        """ 初始化绘制对象 """
        self.button_list: List[draw.Button] = []
        """ 绘制的按钮列表 """
        self.return_list: Dict[str, str] = {}
        """
        按钮返回的响应列表
        按钮返回值:按钮文本
        """
        self.max_width = 0
        """ 绘制的最大宽度 """
        self.max_height = 0
        """ 绘制的最大高度 """

    def set(
        self,
        button_list: List[str],
        return_list: List[str],
        normal_style="standard",
        onbutton_style="onbutton",
    ):
        """
        设置按钮和返回列表
        Keyword arguments:
        button_list -- 按钮列表
        return_list -- 返回列表
        normal_style -- 按钮通常样式
        onbutton_style -- 鼠标悬停时样式
        """
        for i in range(len(button_list)):
            if i <= self.max_height:
                draw_button = draw.Button(button_list[i], return_list[i], normal_style, onbutton_style)
                draw_button.max_width = self.max_width
                self.button_list.append(draw_button)
            self.return_list[return_list[i]] = button_list[i]

    def get_width(self) -> int:
        """
        获取按钮列表的最大宽度
        Return arguments:
        int -- 最大宽度
        """
        return max(self.button_list)

    def get_height(self) -> int:
        """
        获取按钮列表的长度:
        Return arguments:
        int -- 长度
        """
        return len(self.button_list)

    def draw(self):
        """ 绘制按钮列表 """
        for button in self.button_list:
            button.draw()
            io_init.era_print("\n")


class OneMessageAndSingleColumnButton:
    """ 标准单条消息+单列数字id响应按钮监听面板 """

    def __init__(self):
        """ 初始化绘制对象 """
        self.message: draw.RichTextDraw = None
        """ 消息 """
        self.button_panel: SingleColumnButton = None
        """ 按钮监听列表 """

    def set(self, button_list: List[str], message: str, start_id=0):
        """
        设置要绘制的信息
        Keyword arguments:
        button_list -- 监听的按钮列表
        message -- 绘制的消息
        start_id -- id的起始位置
        """
        new_button_list = []
        return_list = []
        for i in range(len(button_list)):
            now_id = id_index(i + start_id)
            now_id_text = now_id + button_list[i]
            new_button_list.append(now_id_text)
            now_i_str = str(start_id + i)
            return_list.append(now_i_str)
        width = normal_config.config_normal.text_width
        self.message = draw.NormalDraw()
        self.message.text = message
        self.message.max_width = width
        self.button_panel = SingleColumnButton()
        self.button_panel.max_width = width
        self.button_panel.max_height = len(return_list)
        self.button_panel.set(new_button_list, return_list)
        self.button_panel.return_list = dict(zip(return_list, button_list))

    def get_return_list(self) -> Dict[str, str]:
        """
        获取按钮响应的id集合
        Return arguments:
        Dict[str,str] -- 按钮响应的id集合 按钮id:按钮文本
        """
        return self.button_panel.return_list

    def draw(self):
        """ 绘制面板 """
        self.message.draw()
        io_init.era_print("\n")
        self.button_panel.draw()


class AskForOneMessage:
    """ 标准单条消息+等待玩家输入任意字符面板 """

    def __init__(self):
        """ 初始化绘制对象 """
        self.message: draw.RichTextDraw = None
        """ 消息 """
        self.input_max: int = 0
        """ 玩家输入文本的最大长度 """

    def set(self, message: str, input_max: int):
        """
        设置要绘制的消息和允许玩家输入的长度
        Keyword arguments:
        message -- 绘制的消息
        input_max -- 允许玩家输入的最大长度
        """
        self.message = draw.NormalDraw()
        self.message.text = message
        self.message.max_width = normal_config.config_normal.text_width
        self.input_max = input_max

    def draw(self) -> str:
        """
        绘制面板
        Return arguments:
        str -- 玩家输入的字符
        """
        return_text = ""
        while 1:
            self.message.draw()
            return_text = flow_handle.askfor_str(1, 1)
            text_index = text_handle.get_text_index(return_text)
            if text_index <= self.input_max:
                break
            io_init.era_print(_(f"输入的字符超长，最大{self.input_max}个英文字符，请重试。"))
            io_init.era_print("\n")
        return return_text


class TitleAndRightInfoListPanel:
    """ 绘制一个标题和一串靠右对齐的列表组合的面板 """

    def __init__(self):
        """ 初始化绘制对象 """
        self.title = ""
        """ 文本标题 """
        self.width = 0
        """ 面板宽度 """
        self.info_list: List[str] = []
        """ 信息列表 """
        self.draw_list: List[draw.NormalDraw] = []
        """ 绘制列表 """

    def set(self, title_text: str, info_list: List[str], width: int):
        """
        设置绘制信息
        Keyword arguments:
        title_text -- 标题
        info_list -- 信息列表
        width -- 标题宽度
        """
        self.width = width
        line = draw.LineDraw("=", width)
        self.draw_list.append(line)
        title = draw.CenterDraw()
        title.max_width = self.width
        title.text = title_text
        self.draw_list.append(title)
        for info_text in info_list:
            info = draw.RightDraw()
            info.max_width = width
            info.text = info_text
            self.draw_list.append(info)
        self.draw_list.append(line)

    def draw(self):
        """ 绘制面板 """
        for value in self.draw_list:
            value.draw()
            io_init.era_print("\n")


class CenterDrawTextListPanel:
    """ 绘制一个列表并居中每个元素 """

    def __init__(self):
        """ 初始化绘制对象 """
        self.width = 0
        """ 面板宽度 """
        self.column = 0
        """ 每行最大元素数 """
        self.draw_list: List[List[draw.CenterDraw]] = []
        """ 绘制列表 """

    def set(self,info_list: List[str],width: int,column: int):
        """
        设置绘制信息
        Keyword arguments:
        info_list -- 信息列表
        width -- 绘制宽度
        column -- 每行最大元素数
        """
        self.width = width
        self.column = column
        new_info_list = value_handle.list_of_groups(info_list,column)
        for now_info_list in new_info_list:
            now_width = int(width / len(now_info_list))
            now_list = []
            for now_info in now_info_list:
                now_info_draw = draw.CenterDraw()
                now_info_draw.max_width = now_width
                now_info_draw.text = now_info
                now_list.append(now_info_draw)
            self.draw_list.append(now_list)

    def draw(self):
        """ 绘制面板 """
        for now_list in self.draw_list:
            for value in now_list:
                value.draw()
            io_init.era_print("\n")


class ClearScreenPanel:
    """ 绘制一屏长度的空行 """

    def draw(self):
        """ 绘制面板 """
        panel = "\n" * 50
        io_init.era_print(panel)


def id_index(now_id: int) -> str:
    """
    生成命令id文本
    Keyword arguments:
    now_id -- 命令id
    Return arguments:
    str -- id文本
    """
    if now_id >= 100:
        return f"[{now_id}]"
    elif now_id >= 10:
        if now_id:
            return f"[0{now_id}]"
        return "[000]"
    return f"[00{now_id}]"
