from types import FunctionType
import datetime
from collections import defaultdict
from operator import attrgetter
from Script.Core import cache_control, game_type, get_text, flow_handle, py_cmd, text_handle
from Script.Design import constant, attr_calculation, attr_text, game_time, character_move
from Script.UI.Moudle import panel, draw
from Script.Config import game_config, normal_config
from Script.UI.Moudle import panel

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


class ClubInfoPanel:
    """
    用于查看社图信息的面板对象
    Keyword arguments:
    width -- 绘制宽度
    """

    def __init__(self, width: int):
        """ 初始化绘制对象 """
        self.width: int = width
        self.now_panel = _("成员")

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
        club_memeber_count = _("人数:") + str(len(club_data.character_set))
        club_theme_info_draw = panel.LeftDrawTextListPanel()
        club_theme_info_draw.set(
            [club_theme_info, club_memeber_count],
            self.width,
            2
        )
        club_owner_info_draw = panel.LeftDrawTextListPanel()
        club_owner_info_draw.set(
            [club_teacher_info, club_owner_info],
            self.width,
            2
        )
        back_draw = draw.CenterButton(_("[返回]"), _("返回"), window_width)
        panel_data = [
            panel.PageHandlePanel(
                list(club_data.character_set),
                SeeCharacterStatusDraw,
                10,
                1,
                window_width,
                0,
                0,
                0
            ),
            SeeClubActivityPanel(club_data.uid,self.width)
        ]
        panel_type_list = [_("成员"), _("活动"), _("其他")]
        while 1:
            if cache.now_panel_id != constant.Panel.VIEW_CLUB_INFO:
                break
            return_list = []
            line_feed.draw()
            title_draw.draw()
            club_theme_info_draw.draw()
            line = draw.LineDraw("+", self.width)
            line.draw()
            club_owner_info_draw.draw()
            line.draw()
            for panel_type in panel_type_list:
                if panel_type == self.now_panel:
                    now_draw = draw.CenterDraw()
                    now_draw.text = f"[{panel_type}]"
                    now_draw.style = "onbutton"
                    now_draw.width = self.width / len(panel_type_list)
                    now_draw.draw()
                else:
                    now_draw = draw.CenterButton(
                        f"[{panel_type}]",
                        panel_type,
                        self.width / len(panel_type_list),
                        cmd_func=self.change_panel,
                        args=(panel_type,),
                    )
                    now_draw.draw()
                    return_list.append(now_draw.return_text)
            line = draw.LineDraw(".", self.width)
            line.draw()
            if self.now_panel == panel_type_list[0]:
                panel_data[0].update()
                panel_data[0].draw()
                return_list.extend(panel_data[0].return_list)
            elif self.now_panel == panel_type_list[1]:
                panel_data[1].draw()
                return_list.extend(panel_data[1].return_list)
            back_draw.draw()
            return_list.append(back_draw.return_text)
            yrn = flow_handle.askfor_all(return_list)
            if yrn == back_draw.return_text:
                cache.now_panel_id = constant.Panel.IN_SCENE
                break

    def change_panel(self, panel_type: str):
        """
        切换当前面板显示的列表类型
        Keyword arguments:
        panel_type -- 要切换的列表类型
        """
        self.now_panel = panel_type
        py_cmd.clr_cmd()


class SeeCharacterStatusDraw:
    """
    社团角色列表中显示角色信息的对象
    Keyword argumentsL
    text -- 角色id
    width -- 最大宽度
    is_button -- 绘制按钮
    num_button -- 绘制数字按钮
    button_id -- 数字按钮id
    """

    def __init__(self, text: int, width: int, _unused: bool, num_button: bool, button_id: int):
        """初始化绘制对象"""
        self.text = text
        """ 角色id """
        self.draw_text: str = ""
        """ 角色状态和名字绘制文本 """
        self.width: int = width
        """ 最大宽度 """
        self.num_button: bool = num_button
        """ 绘制数字按钮 """
        self.button_id: int = button_id
        """ 数字按钮的id """
        self.button_return: str = str(button_id)
        """ 按钮返回值 """
        character_data: game_type.Character = cache.character_data[self.text]
        identity_data: game_type.ClubIdentity = character_data.identity_data[2]
        club_data: game_type.ClubData = cache.all_club_data[identity_data.club_uid]
        cid_text = "No." + str(self.text)
        status_text = game_config.config_status[character_data.state].name
        character_id = character_data.cid
        sex_text = game_config.config_sex_tem[character_data.sex].name
        now_age = character_data.age
        age_text = _("{now_age}岁").format(now_age=now_age)
        draw_skill = -1
        for knowledge in game_config.config_knowledge_type_data[club_data.theme]:
            if knowledge in character_data.knowledge:
                if draw_skill == -1:
                    draw_skill = knowledge
                else:
                    if character_data.knowledge[knowledge] > character_data.knowledge[draw_skill]:
                        draw_skill = knowledge
        self.now_draw_list = []
        cid_text_draw = draw.LeftDraw()
        cid_text_draw.text = cid_text
        cid_text_draw.width = self.width / 10
        self.now_draw_list.append(cid_text_draw)
        character_name_draw = draw.LeftDraw()
        character_name_draw.text = character_data.name
        character_name_draw.width = self.width / 10
        self.now_draw_list.append(character_name_draw)
        sex_text_draw = draw.LeftDraw()
        sex_text_draw.text = sex_text
        sex_text_draw.width = self.width / 10
        self.now_draw_list.append(sex_text_draw)
        age_text_draw = draw.LeftDraw()
        age_text_draw.text = age_text
        age_text_draw.width = self.width / 10
        self.now_draw_list.append(age_text_draw)
        if draw_skill != -1:
            now_skill_text = game_config.config_knowledge[draw_skill].name
            draw_skill_text = _("擅长:") + f"{now_skill_text}"
            skill_info_draw = draw.NormalDraw()
            skill_info_draw.text = draw_skill_text
            skill_info_draw.width = text_handle.get_text_index(now_skill_text)
            skill_level_draw = draw.ExpLevelDraw(character_data.knowledge[draw_skill])
            skill_text_draw = draw.LeftMergeDraw(self.width / 10)
            skill_text_draw.draw_list = [skill_info_draw, skill_level_draw]
            self.now_draw_list.append(skill_text_draw)
        else:
            draw_skill_text = _("擅长:无")
            skill_text_draw = draw.LeftDraw()
            skill_text_draw.text = draw_skill_text
            skill_text_draw.width = self.width / 10
            self.now_draw_list.append(skill_text_draw)
        """ 绘制的对象 """

    def draw(self):
        """绘制对象"""
        for now_draw in self.now_draw_list:
            now_draw.draw()


class SortClubActivityData:
    """
    排序用的活动数据
    Keyword arguments:
    activity_id -- 活动id
    activity_time_id -- 活动时间id
    week -- 活动在周几进行
    start_hour -- 活动开始时间(时)
    start_minute -- 活动开始时间(分)
    """

    def __init__(self, activity_id: str, activity_time_id: str, week: int, start_hour: int, start_minute: int):
        self.activity_id: str = activity_id
        self.activity_time_id: str = activity_time_id
        self.week: int = week
        self.start_hour: int = start_hour
        self.start_minute: int = start_minute


class SeeClubActivityPanel:
    """
    社团活动列表中显示活动信息的面板
    Keyword arguments:
    cid -- 社团id
    width -- 绘制宽度
    """

    def __init__(self, cid: str, width: int):
        self.cid: str = cid
        """ 社团id """
        self.width: int = width
        """ 绘制宽度 """
        self.return_list = []
        if cache.game_time > 0:
            now_date = datetime.datetime.fromtimestamp(cache.game_time, game_time.time_zone)
        else:
            now_date = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=cache.game_time)
        all_activity_list = []
        club_data: game_type.ClubData = cache.all_club_data[self.cid]
        for activity_id in club_data.activity_list:
            activity_data: game_type.ClubActivityData = club_data.activity_list[activity_id]
            for activity_time_id in activity_data.activity_time_list:
                activity_time_data: game_type.ClubActivityTimeData = activity_data.activity_time_list[activity_time_id]
                now_data = SortClubActivityData(
                    activity_data.uid,
                    activity_time_data.uid,
                    activity_time_data.week_day,
                    activity_time_data.start_hour,
                    activity_time_data.start_minute
                )
                all_activity_list.append(now_data)
        grouped_activities = defaultdict(list)
        for activity in all_activity_list:
            grouped_activities[activity.week].append(activity)
        for week in grouped_activities:
            grouped_activities[week].sort(key=attrgetter('start_hour', 'start_minute'))
        self.all_week_panel = {}
        for week in game_config.config_week_day:
            now_list = []
            if week in grouped_activities:
                for now_data in grouped_activities[week]:
                    now_tuple = (cid, now_data.activity_id, now_data.activity_time_id)
                    now_list.append(now_tuple)
            now_panel = panel.PageHandlePanel(now_list,SeeClubActivityDraw,10,1,self.width, 1, 1, 0)
            self.all_week_panel[week] = now_panel
        self.now_week: int = now_date.weekday()

    def draw(self):
        """ 绘制对象 """
        weekday_text_list = [
            game_config.config_week_day[i].name for i in game_config.config_week_day
        ]
        club_data: game_type.ClubData = cache.all_club_data[self.cid]
        now_week_text = game_config.config_week_day[self.now_week].name
        if cache.game_time > 0:
            now_date = datetime.datetime.fromtimestamp(cache.game_time, game_time.time_zone)
        else:
            now_date = datetime.datetime(1970, 1, 1) + datetime.timedelta(seconds=cache.game_time)
        now_date_week = now_date.weekday()
        index = 0
        for week_text in weekday_text_list:
            if week_text == now_week_text:
                if index == now_date_week:
                    week_text = week_text + _("(今日)")
                now_draw = draw.CenterDraw()
                now_draw.text = f"[{week_text}]"
                now_draw.style = "onbutton"
                now_draw.width = self.width / len(weekday_text_list)
                now_draw.draw()
            else:
                if index == now_date_week:
                    week_text = week_text + _("(今日)")
                now_draw = draw.CenterButton(
                    f"[{week_text}]",
                    week_text,
                    self.width / len(weekday_text_list),
                    cmd_func=self.change_now_week,
                    args=(index,),
                )
                now_draw.draw()
                self.return_list.append(now_draw.return_text)
            index += 1
        line_feed.draw()
        line = draw.LineDraw("+", self.width)
        line.draw()
        self.all_week_panel[self.now_week].update()
        self.all_week_panel[self.now_week].draw()
        if len(self.all_week_panel[self.now_week].text_list) <= 10:
            line = draw.LineDraw("-", self.width)
            line.draw()
        self.return_list.extend(self.all_week_panel[self.now_week].return_list)


    def change_now_week(self, week: int):
        """
        切换当前面板显示的星期类型
        Keyword arguments:
        week -- 切换的星期
        """
        self.now_week = week


class SeeClubActivityDraw:
    """
    社团活动列表中显示活动信息的对象
    Keyword argumentsL
    text -- 角色id
    width -- 最大宽度
    is_button -- 绘制按钮
    num_button -- 绘制数字按钮
    button_id -- 数字按钮id
    """

    def __init__(self, text: tuple, width: int, _unused: bool, num_button: bool, button_id: int):
        """初始化绘制对象"""
        self.text = text
        """ 原始数据(社团id,活动唯一id,活动时间唯一id) """
        self.draw_text: str = ""
        """ 绘制文本 """
        self.width: int = width
        """ 最大宽度 """
        self.num_button: bool = num_button
        """ 绘制数字按钮 """
        self.button_id: int = button_id
        """ 数字按钮的id """
        self.button_return: str = str(button_id)
        """ 按钮返回值 """
        index_text = text_handle.id_index(button_id)
        club_data: game_type.ClubData = cache.all_club_data[text[0]]
        activity_data: game_type.ClubActivityData = club_data.activity_list[text[1]]
        self.move_to_position = activity_data.activity_position
        activity_time_data: game_type.ClubActivityTimeData = activity_data.activity_time_list[text[2]]
        start_hour_text = str(activity_time_data.start_hour)
        if len(start_hour_text) == 1:
            start_hour_text = "0" + start_hour_text
        start_minute_text = str(activity_time_data.start_minute)
        if len(start_minute_text) == 1:
            start_minute_text = "0" + start_minute_text
        end_hour_text = str(activity_time_data.end_hour)
        if len(end_hour_text) == 1:
            end_hour_text = "0" + end_hour_text
        end_minute_text = str(activity_time_data.end_minute)
        if len(end_minute_text) == 1:
            end_minute_text = "0" + end_minute_text
        draw_text =  index_text + " " + _("活动:") + f"{activity_data.name} " + _("时间:") + f"{start_hour_text}:{start_minute_text}-{end_hour_text}:{end_minute_text} " + _("地点:") + attr_text.get_scene_path_text(activity_data.activity_position)
        self.now_draw = draw.LeftButton(
            draw_text, self.button_return, self.width, cmd_func=self.move_to_activity_scene
        )

    def draw(self):
        """绘制对象"""
        self.now_draw.draw()

    def move_to_activity_scene(self):
        """ 点击后移动至活动场景 """
        py_cmd.clr_cmd()
        cache.wframe_mouse.w_frame_skip_wait_mouse = 1
        character_move.own_charcter_move(self.move_to_position)

