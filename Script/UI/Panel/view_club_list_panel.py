from types import FunctionType
from Script.UI.Moudle import panel, draw
from Script.Core import (
    game_type, cache_control, text_handle, get_text, py_cmd, flow_handle
)
from Script.Design import constant
from Script.Config import game_config, normal_config

_: FunctionType = get_text._
""" 翻译api """
cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
window_width: int = normal_config.config_normal.text_width
""" 窗体宽度 """
line_feed = draw.NormalDraw()
""" 换行绘制对象 """
line_feed.text = "\n"
line_feed.width = 1


class ClubListPanel:
    """
    用于查看社团列表的面板对象
    Keyword arguments:
    width -- 绘制宽度
    """

    def __init__(self, width: int):
        """ 初始化绘制对象 """
        self.width: int = width

    def draw(self):
        """ 绘制对象 """
        title_draw = draw.TitleLineDraw(_("社团列表"), self.width)
        self.handle_panel = panel.PageHandlePanel(
            list(cache.all_club_data.keys()),
            ClubInfoDraw,
            10,
            1,
            window_width,
            1,
            1,
            0
        )
        while 1:
            if cache.now_panel_id != constant.Panel.VIEW_CLUB_LIST:
                break
            title_draw.draw()
            line = draw.LineDraw("+", self.width)
            line.draw()
            return_list = []
            self.handle_panel.update()
            self.handle_panel.draw()
            return_list.extend(self.handle_panel.return_list)
            line.draw()
            back_draw = draw.CenterButton(_("[返回]"), _("返回"), window_width)
            back_draw.draw()
            line_feed.draw()
            return_list.append(back_draw.return_text)
            yrn = flow_handle.askfor_all(return_list)
            if yrn == back_draw.return_text:
                cache.now_panel_id = constant.Panel.IN_SCENE
                break


class ClubInfoDraw:
    """
    点击后可加入该社团的按钮对象
    Keyword arguments:
    text -- 社团id
    width -- 最大宽度
    is_button -- 绘制按钮
    num_button -- 绘制数字按钮
    button_id -- 数字按钮id
    """

    def __init__(self, text: int, width: int, _unused: bool, num_button: bool, button_id: int):
        """ 初始化绘制对象 """
        self.text = text
        """ 社团id """
        self.draw_text: str = ""
        """ 俱乐部信息绘制文本 """
        self.width: int = width
        """ 最大宽度 """
        self.num_button: bool = num_button
        """ 绘制数字按钮 """
        self.button_id: int = button_id
        """ 数字按钮的id """
        self.button_return: str = str(button_id)
        """ 按钮返回值 """
        club_data: game_type.ClubData = cache.all_club_data[text]
        index_text = text_handle.id_index(button_id)
        club_name = club_data.name
        club_theme = game_config.config_club_theme[club_data.theme].name
        club_theme_text = _("主题:{club_theme}").format(club_theme=club_theme)
        club_member_index = len(club_data.character_set)
        club_member_text = _("人数:{club_member_index}").format(club_member_index=club_member_index)
        teacher_data = cache.character_data[club_data.teacher]
        teacher_name = teacher_data.name
        teacher_name_text = _("指导老师:{teacher_name}").format(teacher_name=teacher_name)
        president_data = cache.character_data[club_data.president]
        president_name = president_data.name
        president_name_text = _("社长:{president_name}").format(president_name=president_name)
        weekend_active_index = 0
        for activity_uid in club_data.activity_list:
            activity_data = club_data.activity_list[activity_uid]
            weekend_active_index += len(activity_data.activity_time_list)
        weekend_active_index_text = _("每周活动次数:{weekend_active_index}").format(weekend_active_index=weekend_active_index)
        button_text = f"{index_text} {club_name} {club_theme_text} {president_name_text} {teacher_name_text} {club_member_text} {weekend_active_index_text}"
        now_draw = draw.LeftButton(button_text, self.button_return, self.width, cmd_func=self.ask_join_club)
        self.now_draw = now_draw

    def draw(self):
        """ 绘制对象 """
        self.now_draw.draw()

    def ask_join_club(self):
        """ 点击后询问是否加入社团 """
        py_cmd.clr_cmd()
        if not self.text:
            return
        now_draw = panel.OneMessageAndSingleColumnButton()
        now_draw.set([_("是"), _("否")], _("想加入这个社团吗?"))
        now_draw.draw()
        return_list = now_draw.get_return_list()
        ans = flow_handle.askfor_all(return_list.keys())
        py_cmd.clr_cmd()
        now_key = return_list[ans]
        if now_key == _("是"):
            py_cmd.clr_cmd()
            line_feed.draw()
            character_data: game_type.Character = cache.character_data[0]
            identity_data = game_type.ClubIdentity()
            identity_data.club_uid = self.text
            character_data.identity_data[identity_data.cid] = identity_data
            club_data = cache.all_club_data[self.text]
            club_data.character_set.add(0)
            cache.now_panel_id = constant.Panel.IN_SCENE
