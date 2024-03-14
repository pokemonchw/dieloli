from Script.UI.Moudle import panel, draw


class ClubListPanel:
    """
    用于查看社团列表的面板对象
    Keyword arguments:
    width -- 绘制宽度
    """

    def __init__(self, width: int):
        """ 初始化绘制对象 """
        self.width: int = width
        self.handle_panel: panel.PageHandlePanel = None
        """ 当前角色列表控制面板 """
