from types import FunctionType
from Script.Core import cache_control, game_type, get_text, flow_handle
from Script.Design import constant
from Script.UI.Moudle import panel, draw
from Script.Config import game_config, normal_config
from Script.UI.Moudle import panel

_: FunctionType = get_text._
""" 翻译api """
cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
line_feed = draw.NormalDraw()
""" 换行绘制对象 """
line_feed.text = "\n"
line_feed.width = 1


class ClubInfoPanel:
    """
    用于查看社图信息的面板对象
    Keyword arguments:
    width -- 绘制宽度
    """

    def __init__(self, width: int):
        """ 初始化绘制对象 """
        self.width: int = width

    def draw(self):
        """ 绘制对象 """
        character_data: game_type.Character = cache.character_data[0]
        if 2 not in character_data.identity_data:
            cache.now_panel_id = constant.Panel.IN_SCENE
            return
        identity_data: game_type.ClubIdentity = character_data.identity_data[2]
        club_data: game_type.ClubData = cache.all_club_data[identity_data.club_uid]
        title_draw = draw.TitleLineDraw(club_data.name, self.width)
        club_owner_data: game_type.Character = cache.character_data[club_data.president]
        club_teacher_data: game_type.Character = cache.character_data[club_data.teacher]
        club_owner_info = _("社长:") + club_owner_data.name
        club_teacher_info = _("指导老师:") + club_teacher_data.name
        club_theme_info = _("主题:") + _(game_config.config_club_theme[club_data.theme].name)
        club_info_draw = panel.LeftDrawTextListPanel()
        club_info_draw.set(
            [club_owner_info, club_teacher_info, club_theme_info],
            self.width,
            3
        )
        while 1:
            line_feed.draw()
            title_draw.draw()
            club_info_draw.draw()
            line = draw.LineDraw("+", self.width)
            line.draw()
            return_list = []
            yrn = flow_handle.askfor_all(return_list)
            if yrn == back_draw.return_text:
                cache.now_panel_id = constant.Panel.IN_SCENE
                break
