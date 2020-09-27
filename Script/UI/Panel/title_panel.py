import gettext
from Script.Config import game_config
from Script.UI.Moudle import panel,draw
from Script.Design import handle_panel

config_normal = game_config.config_normal

@handle_panel.add_panel("title")
def title_panel():
    """ 绘制游戏标题菜单 """
    width = config_normal.text_width
    title_info = panel.TitleAndRightInfoListPanel()
    game_name = config_normal.game_name
    info_list = [config_normal.author,config_normal.verson,config_normal.verson_time]
    title_info.set(config_normal.game_name,info_list,width)
    title_info.draw()
