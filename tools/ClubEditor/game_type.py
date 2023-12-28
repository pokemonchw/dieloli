class ClubData:
    """ 社团数据 """

    def __init__(self):
        self.uid: str = ""
        """ 社团唯一id """
        self.name: str = ""
        """ 社团名 """
        self.premise_data: dict[str, int] = {}
        """ 社团门槛数据 {前提id:占位值} """


class Signal:
    def __init__(self):
        self.__subscribers = []

    def connect(self, func):
        self.__subscribers.append(func)

    def emit(self, *args, **kwargs):
        for subscriber in self.__subscribers:
            subscriber(*args, **kwargs)
