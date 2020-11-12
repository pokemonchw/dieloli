from typing import List
from Script.Core import (
    text_handle,
    io_init,
    rich_text,
    constant,
    py_cmd,
    flow_handle,
)
from Script.Config import game_config

bar_list = set(game_config.config_bar_data.keys())


class NormalDraw:
    """ 通用文本绘制类型 """

    style: str = "standard"
    """ 文本的样式 """

    def __init__(self):
        """ 初始化绘制对象 """
        self.max_width: int = 0
        """ 当前最大可绘制宽度 """
        self.text = ""
        """ 当前要绘制的文本 """

    def __len__(self) -> int:
        """
        获取当前要绘制的文本的长度
        Return arguments:
        int -- 文本长度
        """
        text_index = text_handle.get_text_index(self.text)
        if text_index > self.max_width:
            return self.max_width
        return text_index

    def draw(self):
        """ 绘制文本 """
        if self.__len__() > self.max_width:
            now_text = ""
            if self.max_width > 0:
                for i in self.text:
                    if (
                        text_handle.get_text_index(now_text) + text_handle.get_text_index(i)
                        < self.max_width
                    ):
                        now_text += i
                    break
                now_text = now_text[:-2] + "~"
            io_init.era_print(now_text, self.style)
        else:
            io_init.era_print(self.text, self.style)


class ImageDraw:
    """ 图片绘制 """

    def __init__(self, image_name: str, image_path=""):
        """
        初始化绘制对象
        Keyword arguments:
        image_name -- 图片id
        image_path -- 图片所在路径 (default '')
        """
        self.image_name = image_name
        """ 图片id """
        self.image_path = image_path
        """ 图片所在路径 """
        self.width = 1
        """ 图片宽度 """

    def draw(self):
        """ 绘制图片 """
        io_init.image_print(self.image_name, self.image_path)

    def __len__(self) -> int:
        """ 图片绘制宽度 """
        return self.width


class BarDraw:
    """ 比例条绘制 """

    def __init__(self):
        """ 初始化绘制对象 """
        self.max_width = 0
        """ 比例条最大长度 """
        self.bar_id = ""
        """ 比例条类型id """
        self.draw_list: List[ImageDraw] = []
        """ 比例条绘制对象列表 """

    def set(self, bar_id: str, max_value: int, value: int):
        """
        设置比例条数据
        Keyword arguments:
        bar_id -- 比例条id
        max_value -- 最大数值
        value -- 当前数值
        """
        if self.max_width > 0:
            proportion = 0
            if self.max_width > 1:
                proportion = int(value / max_value * self.max_width)
            fix_bar = int(self.max_width - proportion)
            style_data = game_config.config_bar[game_config.config_bar_data[bar_id]]
            for i in range(proportion):
                now_draw = ImageDraw(style_data.ture_bar, "bar")
                now_draw.width = style_data.width
                self.draw_list.append(now_draw)
            for i in range(fix_bar):
                now_draw = ImageDraw(style_data.null_bar, "bar")
                now_draw.width = style_data.width
                self.draw_list.append(now_draw)

    def draw(self):
        """ 绘制比例条 """
        for bar in self.draw_list:
            bar.draw()

    def __len__(self) -> int:
        """
        获取比例条长度
        Return arguments:
        int -- 比例条长度
        """
        return len(self.draw_list)


class InfoBarDraw:
    """ 带有文本和数值描述的比例条 例: 生命:[图片](2000/2000) """

    def __init__(self):
        """ 初始化绘制对象 """
        self.max_width: int = 0
        """ 比例条最大长度 """
        self.bar_id: str = ""
        """ 比例条类型id """
        self.text: str = ""
        """ 比例条描述文本 """
        self.draw_list: List[ImageDraw] = []
        """ 比例条绘制对象列表 """
        self.scale: float = 1
        """ 比例条绘制区域占比 """

    def set(self, bar_id: str, max_value: int, value: int, text: str):
        """
        设置比例条数据
        Keyword arguments:
        bar_id -- 比例条id
        max_value -- 最大数值
        value -- 当前数值
        text -- 描述文本
        """
        now_max_width = int(self.max_width * self.scale)
        info_draw = NormalDraw()
        info_draw.max_width = int(now_max_width / 3)
        info_draw.text = f"{text}["
        value_draw = NormalDraw()
        value_draw.max_width = int(now_max_width / 3)
        value_draw.text = f"(]{value}/{max_value})"
        self.bar_id = bar_id
        bar_draw = BarDraw()
        bar_draw.max_width = now_max_width - len(info_draw) - len(value_draw)
        bar_draw.set(self.bar_id, max_value, value)
        fix_width = int((self.max_width - now_max_width) / 2)
        fix_draw = NormalDraw()
        fix_draw.text = " " * fix_width
        fix_draw.max_width = fix_width
        self.draw_list = [fix_draw, info_draw, bar_draw, value_draw, fix_draw]

    def draw(self):
        """ 绘制比例条 """
        for bar in self.draw_list:
            bar.draw()


class Button:
    """ 按钮绘制 """

    def __init__(
        self,
        text: str,
        return_text: str,
        normal_style="standard",
        on_mouse_style="onbutton",
    ):
        """
        初始化绘制对象
        Keyword arguments:
        text -- 按钮文本
        return_text -- 点击按钮响应文本
        normal_style -- 按钮默认样式
        on_mouse_style -- 鼠标悬停时样式
        """
        self.text: str = text
        """ 按钮文本 """
        self.return_text: str = return_text
        """ 点击按钮响应文本 """
        self.normal_style: str = normal_style
        """ 按钮默认样式 """
        self.on_mouse_style: str = on_mouse_style
        """ 鼠标悬停时样式 """
        self.max_width: int = 0
        """ 按钮文本的最大宽度 """

    def __len__(self) -> int:
        """
        获取按钮文本长度
        Return arguments:
        int -- 文本长度
        """
        return text_handle.get_text_index(self.text)

    def __lt__(self, other):
        """
        比较两个button对象的文本长度
        Keyword arguments:
        other -- 要比较的button对象
        Return arguments:
        bool -- 大小校验
        """
        return len(self) < len(other)

    def draw(self):
        """ 绘制按钮 """
        if self.max_width < len(self):
            now_text = ""
            if self.max_width > 0:
                for i in self.text:
                    if (
                        text_handle.get_text_index(now_text) + text_handle.get_text_index(i)
                        < self.max_width
                    ):
                        now_text += i
                        continue
                    break
                now_text = now_text[:-2] + "~"
            py_cmd.pcmd(
                now_text,
                self.return_text,
                normal_style=self.normal_style,
                on_style=self.on_mouse_style,
            )
        else:
            py_cmd.pcmd(
                self.text,
                self.return_text,
                normal_style=self.normal_style,
                on_style=self.on_mouse_style,
            )


class CenterButton:
    """ 居中按钮绘制 """

    def __init__(self,text:str,return_text:str,width:int,fix_text=" ",normal_style="standard",on_mouse_style="onbutton"):
        """
        初始化绘制对象
        Keyword arguments:
        text -- 按钮原始文本
        return_text -- 点击按钮响应文本
        width -- 按钮最大绘制宽度
        fix_text -- 对齐用补全文本
        normal_style -- 按钮默认样式
        on_mouse_style -- 鼠标悬停时样式
        """
        self.text: str = text
        """ 按钮文本 """
        self.return_text: str = return_text
        """ 点击按钮响应文本 """
        self.fix_text: str = fix_text
        """ 对齐用补全文本 """
        self.normal_style: str = normal_style
        """ 按钮默认样式 """
        self.on_mouse_style: str = on_mouse_style
        """ 鼠标悬停时样式 """
        self.max_width: str = width
        """ 按钮文本的最大宽度 """

    def __len__(self) -> int:
        """
        获取按钮文本长度
        Return arguments:
        int -- 文本长度
        """
        return self.max_width

    def __it__(self,other:Button):
        """
        比较两个button对象的文本长度
        Keyword arguments:
        other -- 要比较的button对象
        Return arguments:
        bool -- 大小校验
        """
        return self.max_width < other.max_width

    def draw(self):
        """ 绘制按钮 """
        print(self.max_width)
        if self.max_width < text_handle.get_text_index(self.text):
            now_text = ""
            print(text_handle.get_text_index(self.text))
            print(self.text)
            if self.max_width > 0:
                for i in self.text:
                    if text_handle.get_text_index(now_text) + text_handle.get_text_index(i) < self.max_width:
                        now_text += i
                    break
                now_text = now_text[:-2] + "~"
        else:
            now_index = text_handle.get_text_index(self.text)
            now_text = self.text
            if now_index == self.max_width - 1:
                now_text = " " + now_text
            elif now_index == self.max_width - 2:
                now_text = " " + now_text + " "
            else:
                now_text = text_handle.align(now_text,"center",0,1,self.max_width)
        py_cmd.pcmd(
            now_text,
            self.return_text,
            normal_style=self.normal_style,
            on_style=self.on_mouse_style,
        )


class LineDraw:
    """ 绘制线条文本 """

    def __init__(self, text: str, width: int, style="standard"):
        """
        初始化绘制对象
        Keyword arguments:
        text -- 用于绘制线条的文本
        width -- 线条宽度
        style -- 当前默认文本样式
        """
        self.text = text
        """ 用于绘制线条的文本 """
        self.style = style
        """ 文本样式 """
        self.width = width
        """ 线条宽度 """

    def __len__(self) -> int:
        """
        获取当前要绘制的文本的长度
        Return arguments:
        int -- 文本长度
        """
        return self.width

    def draw(self):
        """ 绘制线条 """
        text_index = text_handle.get_text_index(self.text)
        text_num = self.width / text_index
        now_draw = NormalDraw()
        now_draw.max_width = self.width
        now_draw.text = self.text * int(text_num) + "\n"
        now_draw.style = self.style
        now_draw.draw()


class TitleLineDraw:
    """ 绘制标题线文本 """

    def __init__(
        self,
        title: str,
        width: int,
        line: str = "=",
        frame="口",
        style="standard",
        title_style="littletitle",
        frame_style="littletitle",
    ):
        """
        初始化绘制对象
        Keyword arguments:
        text -- 标题
        width -- 标题线线条宽度
        line -- 用于绘制线条的文本
        frame -- 用于绘制标题边框的文本
        style -- 线条样式
        title_style -- 标题样式
        frame_style -- 标题边框样式
        """
        self.title = title
        """ 标题 """
        self.width = width
        """ 线条宽度 """
        self.line = line
        """ 用于绘制线条的文本 """
        self.frame = frame
        """ 用于绘制标题边框的文本 """
        self.style = style
        """ 线条默认样式 """
        self.title_style = title_style
        """ 标题样式 """
        self.frame_style = frame_style
        """ 标题边框样式 """

    def draw(self):
        """ 绘制线条 """
        title_draw = NormalDraw()
        title_draw.max_width = self.width
        title_draw.style = self.title_style
        title_draw.text = f" {self.title} "
        fix_width = self.width - len(title_draw)
        if fix_width < 0:
            fix_width = 0
        frame_width = int(fix_width / 2)
        frame_draw = NormalDraw()
        frame_draw.max_width = frame_width
        frame_draw.style = self.frame_style
        frame_draw.text = self.frame
        line_width = int(fix_width / 2 - len(frame_draw))
        if line_width < 0:
            line_width = 0
        line_draw = NormalDraw()
        line_draw.max_width = line_width
        line_draw.style = self.style
        line_draw.text = self.line * line_width
        for text in [line_draw, frame_draw, title_draw, frame_draw, line_draw]:
            text.draw()
        io_init.era_print("\n")


class LittleTitleLineDraw:
    """ 绘制小标题线文本 """

    def __init__(self,title: str,width: int,line:str="=",style="standard",title_style="sontitle"):
        """
        初始化绘制对象
        Keyword arguments:
        title -- 标题
        width -- 标题线线条宽度
        line -- 用于绘制线条的文本
        style -- 线条样式
        title_style -- 标题样式
        """
        self.title = title
        """ 标题 """
        self.width = width
        """ 线条宽度 """
        self.line = line
        """ 用于绘制线条的文本 """
        self.style = style
        """ 线条默认样式 """
        self.title_style = title_style
        """ 标题样式 """

    def draw(self):
        """ 绘制线条 """
        title_draw = NormalDraw()
        title_draw.max_width = self.width
        title_draw.text = self.title
        title_draw.style = self.title_style
        line_a_width = int(self.width / 4) - len(title_draw)
        if line_a_width < 0:
            line_a_width = 0
        line_a = NormalDraw()
        line_a.max_width = line_a_width
        line_a.style = self.style
        line_a.text = self.line * line_a_width
        line_b = NormalDraw()
        line_b.max_width = self.width - len(title_draw) - len(line_a)
        line_b.style = self.style
        line_b.text = self.line * line_b.max_width
        for value in [line_a,title_draw,line_b]:
            value.draw()
        io_init.era_print("\n")

class CenterDraw(NormalDraw):
    """ 居中绘制文本 """

    def draw(self):
        """ 绘制文本 """
        self.max_width = int(self.max_width)
        if len(self) > self.max_width:
            now_text = ""
            if self.max_width > 0:
                for i in self.text:
                    if (
                        text_handle.get_text_index(now_text) + text_handle.get_text_index(i)
                        < self.max_width
                    ):
                        now_text += i
                    break
                now_text = now_text[:-2] + "~"
            io_init.era_print(now_text, self.style)
        elif len(self) > self.max_width - 1:
            now_text = " " + self.text
        elif len(self) > self.max_width - 2:
            now_text = " " + self.text + " "
        else:
            now_text = text_handle.align(self.text, "center", 0, 1, self.max_width)
        if len(self) < self.max_width:
            now_text += " " * (int(self.max_width) - text_handle.get_text_index(now_text))
        io_init.era_print(now_text, self.style)


class RightDraw(NormalDraw):
    """ 右对齐绘制文本 """

    def draw(self):
        """ 绘制文本 """
        if len(self) > self.max_width:
            now_text = ""
            if self.max_width > 0:
                for i in self.text:
                    if (
                        text_handle.get_text_index(now_text) + text_handle.get_text_index(i)
                        < self.max_width
                    ):
                        now_text += i
                    break
                now_text = now_text[:-2] + "~"
        elif len(self) > self.max_width - 2:
            now_text = " " + self.text
        else:
            now_text = text_handle.align(self.text, "right", 0, 1, self.max_width)
        io_init.era_print(now_text, self.style)
