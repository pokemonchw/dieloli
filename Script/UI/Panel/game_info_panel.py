from types import FunctionType
from Script.UI.Moudle import draw
from Script.Design import game_time, character, course
from Script.Core import get_text, cache_control, game_type
from Script.Config import game_config

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
_: FunctionType = get_text._
""" 翻译api """


class GameTimeInfoPanel:
    """
    查看游戏时间面板
    Keyword arguments:
    width -- 最大宽度
    """

    def __init__(self, width: int):
        """初始化绘制对象"""
        self.width = width
        """ 面板的最大宽度 """
        now_width = 0
        now_draw = draw.CenterMergeDraw(self.width)
        date_draw = draw.NormalDraw()
        date_draw.width = self.width
        date_draw.text = f"{game_time.get_date_text()} {game_time.get_week_day_text()} "
        now_draw.draw_list.append(date_draw)
        now_width += len(date_draw)
        solar_period = game_time.get_solar_period(cache.game_time)
        season = game_config.config_solar_period[solar_period].season
        season_config = game_config.config_season[season]
        season_draw = draw.NormalDraw()
        season_draw.text = f"{season_config.name} "
        season_draw.style = "season"
        season_draw.width = self.width - now_width
        now_draw.draw_list.append(season_draw)
        now_width += len(season_draw)
        judge, solar_period = game_time.judge_datetime_solar_period(cache.game_time)
        if judge:
            solar_period_config = game_config.config_solar_period[solar_period]
            solar_period_draw = draw.NormalDraw()
            solar_period_draw.text = f"{solar_period_config.name} "
            solar_period_draw.width = self.width - now_width
            solar_period_draw.style = "solarperiod"
            now_draw.draw_list.append(solar_period_draw)
            now_width += len(solar_period_draw)
        sun_time = game_time.get_sun_time(cache.game_time)
        sun_time_config = game_config.config_sun_time[sun_time]
        sun_time_draw = draw.NormalDraw()
        sun_time_draw.text = f"{sun_time_config.name} "
        sun_time_draw.width = self.width - now_width
        now_draw.draw_list.append(sun_time_draw)
        now_width += len(sun_time_draw)
        if sun_time <= 2 or sun_time >= 10:
            moon_phase = game_time.get_moon_phase(cache.game_time)
            moon_phase_config = game_config.config_moon[moon_phase]
            moon_phase_draw = draw.NormalDraw()
            moon_phase_draw.text = f"{moon_phase_config.name} "
            moon_phase_draw.width = self.width - now_width
            moon_phase_draw.style = "moon"
            now_draw.draw_list.append(moon_phase_draw)
            now_width += len(moon_phase_draw)
        now_judge = game_time.judge_attend_class_today(0)
        attend_class = _("(休息)")
        if now_judge:
            attend_class = _("(上学)")
        attend_class += " "
        attend_class_draw = draw.NormalDraw()
        attend_class_draw.text = attend_class
        attend_class_draw.width = self.width - now_width
        now_draw.draw_list.append(attend_class_draw)
        now_width += len(attend_class_draw)
        now_course_id = -1
        attend_class_judge, school_id, now_week, session_id, now_course_id = character.judge_character_in_class_time(0)
        if attend_class_judge:
            character_data: game_type.Character = cache.character_data[0]
            if character_data.age <= 18:
                _unused, phase = course.get_character_school_phase(0)
                cache.course_time_table_data[school_id].setdefault(phase,{})
                cache.course_time_table_data[school_id][phase].setdefault(now_week,{})
                if session_id in cache.course_time_table_data[school_id][phase][now_week]:
                    now_course_id = cache.course_time_table_data[school_id][phase][now_week][session_id]
            now_course_text = ""
            if not session_id:
                now_course_text = _("早读课")
            elif now_course_id == -1:
                now_course_text = _("自习课")
            else:
                now_course_text = game_config.config_course[now_course_id].name
            now_course_text = f"({now_course_text})"
            now_course_text += " "
            course_draw = draw.NormalDraw()
            course_draw.text = now_course_text
            course_draw.width = self.width - now_width
            now_draw.draw_list.append(course_draw)
            now_width += len(course_draw)
        self.width = now_width
        now_draw.width = self.width
        self.now_draw: draw.NormalDraw = now_draw
        """ 当前面板绘制对象 """

    def __len__(self):
        """获取绘制宽度"""
        return self.width

    def draw(self):
        """绘制对象"""
        self.now_draw.draw()
