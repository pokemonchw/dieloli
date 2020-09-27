from typing import List
from Script.Core import era_print,text_handle,io_init,rich_text,constant,py_cmd,flow_handle
from Script.Config import game_config

bar_list = set(game_config.config_bar_data.keys())


class NormalDraw:
    """ 通用文本绘制类型 """

    def __init__(self):
        """ 初始化绘制对象 """
        self.max_width:int = 0
        """ 当前最大可绘制宽度 """
        self.text = ""
        """ 当前要绘制的文本 """
        self.style = "standard"
        """ 文本的样式 """

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
                    if text_handle.get_text_index(now_text) + text_handle.get_text_index(i) < self.max_width:
                        now_text += i
                    break
                now_text[len(now_text) - 1] = "~"
            io_init.era_print(now_text,self.style)
        else:
            io_init.era_print(self.text,self.style)


class ImageDraw:
    """ 图片绘制 """

    def __init__(self,image_name:str,image_path=""):
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
        self.bar_text = ""
        """ 比例条原始文本 """
        self.draw_list:List[ImageDraw] = []
        """ 比例条绘制对象列表 """

    def set(self,bar_id:str,max_value:int,value:int):
        """ 设置比例条数据 """
        if self.max_width > 0:
            proportion = 0
            if self.max_width > 1:
                proportion = int(value / max_value * self.max_width)
            fix_bar = self.max_width - proportion
            style_data = game_config.config_bar[game_config.config_bar_data[bar_id]]
            for i in range(proportion):
                now_draw = ImageDraw()
                now_draw.image_name = style_data.ture_bar
                now_draw.width = style_data.width
                self.draw_list.append(now_draw)
            for i in range(fix_bar):
                now_draw = ImageDraw()
                now_draw.image_name = style_data.null_bar
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


class RichTextDraw(NormalDraw):
    """ 富文本绘制 """

    def __init__(self):
        """ 初始化绘制对象 """
        self.original_text:str = ""
        """ 原始文本 """
        self.style_list:List[str] = []
        """ 文本样式列表 """
        self.draw_list:List[NormalDraw] = []
        """ 绘制对象列表 """

    def draw(self):
        """ 绘制文本 """
        for text in self.draw_list:
            text.draw()

    def set_text(self,origin_text:str):
        """
        设置原始文本
        Keyword arguments:
        origin_text -- 原始文本
        """
        self.style_list = rich_text.get_rich_text_print(origin_text,self.style)
        self.text = rich_text.remove_rich_cache(origin_text)
        now_index = 0
        now_width = 0
        while 1:
            if now_index >= len(self.text):
                break
            now_text = self.text[now_index]
            goto_index = now_index + 1
            if now_index < len(self.text) - 1:
                for i in range(now_index+1,len(self.text)):
                    if self.style_list[i] != style_list[now_index]:
                        break
                    now_text += self.text[i]
                    goto_index = i + 1
                new_draw = NormalDraw()
                new_draw.style = style_list[now_index]
                new_draw.text = now_text
                text_width = text_handle.get_text_index(now_text)
                if now_width + text_width > self.max_width:
                    if self.max_width - now_width > 0:
                        new_draw.max_width = self.max_width - now_width
                else:
                    new_draw.max_width = text_width
                    now_width += text_width
                self.draw_list.append(new_draw)
                now_index = goto_index


class Button:
    """ 按钮绘制 """

    def __init__(self,text:str,return_text:str,normal_style="standard",on_mouse_style="onbutton"):
        """
        初始化绘制对象
        Keyword arguments:
        text -- 按钮文本
        return_text -- 点击按钮响应文本
        normal_style -- 按钮默认样式
        on_mouse_style -- 鼠标悬停时样式
        """
        self.text:str = ""
        """ 按钮文本 """
        self.return_text:str = ""
        """ 点击按钮响应文本 """
        self.normal_style = normal_style
        """ 按钮默认样式 """
        self.on_mouse_style = on_mouse_style
        """ 鼠标悬停时样式 """
        self.max_width = 0
        """ 按钮文本的最大宽度 """

    def __len__(self) -> int:
        """
        获取按钮文本长度
        Return arguments:
        int -- 文本长度
        """
        return text_handle.get_text_index(self.text)

    def __lt__(self,other):
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
                    if text_handle.get_text_index(now_text) + text_handle.get_text_index(i) < self.max_width:
                        now_text += i
                    break
                now_text[len(now_text) - 1] = "~"
            py_cmd.pcmd(now_text,self.return_text,normal_style=self.normal_style,on_style=self.on_mouse_style)
        else:
            py_cmd.pcmd(self.text,self.return_text,normal_style=self.normal_style,on_style=self.on_mouse_style)


class LineDraw:
    """ 绘制线条文本 """

    def __init__(self,text:str,width:int,style="standard"):
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
        io_init.era_print(self.text*int(text_num),self.style)


class CenterDraw(NormalDraw):
    """ 剧中绘制文本 """

    def draw(self):
        """ 绘制文本 """
        if len(self) > self.max_width:
            now_text = ""
            if self.max_width > 0:
                for i in self.text:
                    if text_handle.get_text_index(now_text) + text_handle.get_text_index(i) < self.max_width:
                        now_text += i
                    break
                now_text[len(now_text) - 1] = "~"
            io_init.era_print(now_text,self.style)
        elif len(self) > self.max_width - 1:
            now_text = " " + self.text
        elif len(self) > self.max_width - 2:
            now_text = " " + self.text + " "
        else:
            now_text = text_handle.align(self.text,"center",0,1,self.max_width)
        io_init.era_print(now_text,self.style)


class RightDraw(NormalDraw):
    """ 右对齐绘制文本 """

    def draw(self):
        """ 绘制文本 """
        if len(self) > self.max_width:
            now_text = ""
            if self.max_width > 0:
                for i in self.text:
                    if text_handle.get_text_index(now_text) + text_handle.get_text_index(i) < self.max_width:
                        now_text += i
                    break
                now_text[len(now_text) - 1] = "~"
        elif len(self) > self.max_width - 2:
            now_text = " " + self.text
        else:
            now_text = text_handle.align(self.text,"right",0,1,self.max_width)
        io_init.era_print(now_text,self.style)
