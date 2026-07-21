from typing import List, Dict


class ClubData:
    """ 社团数据 """

    def __init__(self):
        self.uid: str = ""
        """ 社团唯一id """
        self.name: str = ""
        """ 社团名 """
        self.theme: int = 0
        """ 社团主题 """
        self.premise_data: Dict[str, int] = {}
        """ 社团门槛数据 {前提id:占位值} """
        self.activity_list: Dict[str, ClubActivityData] = {}
        """ 社团活动数据 {活动唯一id:活动数据} """


class ClubActivityData:
    """ 社团活动数据 """

    def __init__(self):
        self.uid: str = ""
        """ 活动唯一id """
        self.name: str = ""
        """ 活动名 """
        self.activity_time_list: Dict[str, ClubActivityTimeData] = {}
        """ 活动时间 {活动时间唯一id:活动时间数据} """
        self.activity_position: List[str] = []
        """ 活动地点 """
        self.description: int = 0
        """ 活动内容 """


class ClubActivityTimeData:
    """ 社团活动时间数据 """

    def __init__(self):
        self.uid: str = ""
        """ 活动时间唯一id """
        self.week_day: int = 0
        """ 周几 """
        self.start_hour: int = 0
        """ 起始时间(时) """
        self.start_minute: int = 0
        """ 起始时间(分) """
        self.end_hour: int = 0
        """ 结束时间(时) """
        self.end_minute: int = 0
        """ 结束时间(分) """


class Signal:
    def __init__(self):
        self.__subscribers = []

    def connect(self, func):
        self.__subscribers.append(func)

    def emit(self, *args, **kwargs):
        for subscriber in self.__subscribers:
            subscriber(*args, **kwargs)



class Map:
    """地图数据"""

    def __init__(self):
        self.map_path: str = ""
        """ 地图路径 """
        self.map_name: str = ""
        """ 地图名字 """
        self.path_edge: Dict[str, Dict[str, int]] = {}
        """
        地图下场景通行路径
        场景id:可直达场景id:移动所需时间
        """
        self.map_draw: MapDraw = MapDraw()
        """ 地图绘制数据 """
        self.sorted_path: Dict[str, Dict[str, TargetPath]] = {}
        """
        地图下场景间寻路路径
        当前节点:目标节点:路径对象
        """


class MapDraw:
    """地图绘制数据"""

    def __init__(self):
        self.draw_text: List[MapDrawLine] = []
        """ 绘制行对象列表 """


class MapDrawLine:
    """地图绘制行数据"""

    def __init__(self):
        self.width: int = 0
        """ 总行宽 """
        self.draw_list: List[MapDrawText] = []
        """ 绘制的对象列表 """


class MapDrawText:
    """地图绘制文本数据"""

    def __init__(self):
        self.text: str = ""
        """ 要绘制的文本 """
        self.is_button: bool = 0
        """ 是否是场景按钮 """


class TargetPath:
    """寻路目标路径数据"""

    def __init__(self):
        self.path: List[str] = []
        """ 寻路路径节点列表 """
        self.time: List[int] = []
        """ 移动所需时间列表 """


class Scene:
    """场景数据"""

    def __init__(self):
        self.scene_path: str = ""
        """ 场景路径 """
        self.scene_name: str = ""
        """ 场景名字 """
        self.in_door: bool = 0
        """ 在室内 """
        self.scene_tag: str = ""
        """ 场景标签 """
        self.character_list: set = set()
        """ 场景内角色列表 """
