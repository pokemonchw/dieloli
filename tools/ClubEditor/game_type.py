from typing import List, Dict


class ClubData:
    """ 社团数据 """

    def __init__(self):
        self.uid: str = ""
        """ 社团唯一id """
        self.name: str = ""
        """ 社团名 """
        self.theme: str = ""
        """ 社团主题 """
        self.premise_data: Dict[str, int] = {}
        """ 社团门槛数据 {前提id:占位值} """
        self.activity_list: Dict[str, ClubActivityData] = []
        """ 社团活动数据 {活动唯一id:活动数据} """


class ClubActivityData:
    """ 社团活动数据 """

    def __init__(self):
        self.uid: str = ""
        """ 活动唯一id """
        self.name: str = ""
        """ 活动名 """
        self.activity_time_list: Dict[str, ClubActivityData] = []
        """ 活动时间 {活动时间唯一id:活动时间数据} """
        self.activity_position: List[str] = []
        """ 活动地点 """


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
