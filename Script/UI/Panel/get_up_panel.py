from types import FunctionType
from Script.UI.Moudle import draw,panel
from Script.UI.Panel import see_character_info_panel,game_info_panel,see_save_info_panel
from Script.Design import game_time
from Script.Core import get_text, cache_control, flow_handle, py_cmd, text_handle,game_type
from Script.Config import game_config
import time

cache:game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
_: FunctionType = get_text._
""" 翻译api """
line_feed = draw.NormalDraw()
""" 换行绘制对象 """
line_feed.text = "\n"
line_feed.width = 1

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
            line_feed.draw()
            title_draw = draw.TitleLineDraw(_("主页"), self.width)
            character_data = cache.character_data[self.character_id]
            title_draw.draw()
            game_time_draw = game_info_panel.GameTimeInfoPanel(self.width / 2)
            game_time_draw.now_draw.width = len(game_time_draw)
            game_time_draw.draw()
            line_feed.draw()
            line_feed.draw()
            player_info_draw = see_character_info_panel.CharacterInfoHead(0,self.width)
            player_info_draw.draw_title = 0
            player_info_draw.draw()
            line_feed.draw()
            game_menu_titie = draw.LittleTitleLineDraw(_("游戏菜单"),self.width)
            game_menu_titie.draw()
            get_up_button = draw.CenterButton(_("[000]睁眼起床"),"0",self.width / 3)
            get_up_button.draw()
            see_character_list_button = draw.CenterButton(_("[001]查看属性"),"1",self.width/3,cmd_func=self.see_character_list)
            see_character_list_button.draw()
            save_button = draw.CenterButton(_("[002]读写存档"),"2",self.width / 3,cmd_func=self.see_save_handle)
            save_button.draw()
            return_list = [get_up_button.return_text,see_character_list_button.return_text,save_button.return_text]
            yrn = flow_handle.askfor_all(return_list)

    def see_character_list(self):
        """ 绘制角色列表 """
        py_cmd.clr_cmd()
        line_feed.draw()
        title_draw = draw.TitleLineDraw(_("角色列表"),self.width)
        handle_panel = panel.PageHandlePanel(list(cache.character_data.keys()),see_character_info_panel.GetUpCharacterInfoDraw,10,1,self.width,1,1,0,"-")
        while 1:
            title_draw.draw()
            self.return_list = []
            handle_panel.update()
            handle_panel.draw()
            self.return_list.extend(handle_panel.return_list)
            back_draw = draw.CenterButton(_("[返回]"),_("返回"),self.width)
            back_draw.draw()
            self.return_list.append(back_draw.return_text)
            yrn = flow_handle.askfor_all(self.return_list)
            py_cmd.clr_cmd()
            if yrn == back_draw.return_text:
                break

    def see_save_handle(self):
        """ 处理读写存档 """
        now_panel = see_save_info_panel.SeeSaveListPanel(self.width,1)
        now_panel.draw()

