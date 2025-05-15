from typing import Dict


class Signal:
    def __init__(self):
        self.__subscribers = []

    def connect(self, func):
        self.__subscribers.append(func)

    def emit(self, *args, **kwargs):
        for subscriber in self.__subscribers:
            subscriber(*args, **kwargs)


class ClothingTem:
    """ 服装模板 """

    cid: str
    """ 模板id """
    name: str
    """ 服装名字 """
    clothing_type: int
    """ 服装类型 """
    sex: int
    """ 服装性别限制 """
    tag: int
    """ 服装用途标签 """
    describe: str
    """ 描述 """

    def __init__(self):
        self.cid = ""
        self.name = ""
        self.clothing_type = 0
        self.sex = 0
        self.tag = 0
        self.describe = ""


class ClothingSuit:
    """ 套装配置数据 """

    cid: str
    """ 套装id """
    name: str
    """ 套装名称 """
    clothing_wear: Dict[int, str]
    """ 服装设置 {穿戴位置:服装id} """

    def __init__(self):
        self.cid = ""
        self.name = ""
        self.clothing_wear = {}
