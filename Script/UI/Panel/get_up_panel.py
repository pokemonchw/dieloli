from types import FunctionType
from Script.UI.Moudle import draw
from Script.UI.Panel import see_character_info_panel
from Script.Design import game_time
from Script.Core import get_text, cache_contorl, flow_handle, py_cmd
from Script.Config import game_config
import time

_: FunctionType = get_text._
""" 翻译api """

class GetUpPanel:
    """
    用于查看角色起床界面面板对象
    Keyword arguments:
    character_id -- 角色id
    width -- 绘制宽度
    """

    def __init__(self,character_id:int,width:int):
        """ 初始化绘制对象 """
        self.width:int = width
        """ 绘制的最大宽度 """
        self.character_id:int = character_id
        """ 要绘制的角色id """

    def draw(self):
        """ 绘制面板 """
        while 1:
            title_draw = draw.TitleLineDraw(_("主页"), self.width)
            character_data = cache_contorl.character_data[self.character_id]
            title_draw.draw()
            now_width = 0
            now_draw = draw.CenterMergeDraw(self.width)
            date_draw = draw.NormalDraw()
            date_draw.width = self.width
            date_draw.text = f"{game_time.get_date_text()} {game_time.get_week_day_text()} "
            now_draw.draw_list.append(date_draw)
            now_width += len(date_draw)
            solar_period = game_time.get_solar_period(cache_contorl.game_time)
            season = game_config.config_solar_period[solar_period].season
            season_config = game_config.config_season[season]
            season_draw = draw.NormalDraw()
            season_draw.text = f"{season_config.name} "
            season_draw.style = "season"
            season_draw.width = self.width - now_width
            now_draw.draw_list.append(season_draw)
            now_width += len(season_draw)
            judge,solar_period = game_time.judge_datetime_solar_period(cache_contorl.game_time)
            if judge:
                solar_period_config = game_config.config_solar_period[solar_period]
                solar_period_draw = draw.NormalDraw()
                solar_period_draw.text = f"{solar_period_config.name} "
                solar_period_draw.width = self.width - now_width
                solar_period_draw.style = "solarperiod"
                now_draw.draw_list.append(solar_period_draw)
                now_width += len(solar_period_draw)
            name_draw = draw.Button(character_data.name,character_data.name,cmd_func=self.see_character)
            name_draw.width = self.width - now_width
            now_draw.draw_list.append(name_draw)
            now_width += len(name_draw)
            gold_draw = draw.NormalDraw()
            gold_draw.width = self.width - now_width
            gold_draw.text = f" 现金:{character_data.gold}$"
            now_draw.draw_list.append(gold_draw)
            now_draw.draw()
            yrn = flow_handle.askfor_all([name_draw.return_text])

    def see_character(self):
        """ 绘制角色属性 """
        py_cmd.clr_cmd()
        attr_panel = see_character_info_panel.SeeCharacterInfoOnGetUpPanel(self.character_id,self.width)
        attr_panel.draw()
