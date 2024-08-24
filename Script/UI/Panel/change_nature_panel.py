from typing import Dict
from types import FunctionType
from Script.Core import cache_control, game_type, value_handle, flow_handle, get_text
from Script.Config import game_config
from Script.UI.Moudle import draw, panel

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
_: FunctionType = get_text._
""" 翻译api """
line_feed_draw = draw.NormalDraw()
""" 绘制换行对象 """
line_feed_draw.text = "\n"

class ChangeNaturePanel:
    """
    修改角色性格面板对象
    Keyword arguments:
    width -- 绘制宽度
    """

    def __init__(self, width: int):
        """ 初始化绘制对象 """
        self.width: int = width
        """ 绘制的最大宽度 """
        player_data: game_type.Character = cache.character_data[0]
        self.nature: Dict[int, int] = player_data.nature.copy()
        """ 当前的角色性格数据 """

    def draw(self):
        """ 绘制对象 """
        while 1:
            line_feed_draw.draw()
            title_draw = draw.TitleLineDraw(_("人物性格"), self.width)
            title_draw.draw()
            ask_list = []
            self.draw_list = []
            for nature_type in game_config.config_nature_tag:
                type_config = game_config.config_nature_tag[nature_type]
                nature_set = game_config.config_nature_tag_data[nature_type]
                type_value = 0
                nature_draw_list = []
                nature_group = value_handle.list_of_groups(list(nature_set), 1)
                for nature_list in nature_group:
                    for nature_id in nature_list:
                        nature_config = game_config.config_nature[nature_id]
                        nature_value = 0
                        if nature_id in self.nature:
                            nature_value = self.nature[nature_id]
                        type_value += nature_value
                        good_judge = False
                        if nature_value >= 50:
                            good_judge = True
                        nature_draw = None
                        nature_width = int(self.width / len(nature_group))
                        if good_judge:
                            nature_draw = draw.CenterButton(nature_config.good, nature_config.good, nature_width, cmd_func=self.change_nature, args=(nature_id,))
                        else:
                            nature_draw = draw.CenterButton(nature_config.bad, nature_config.bad, nature_width, cmd_func=self.change_nature, args=(nature_id,))
                        ask_list.append(nature_draw.return_text)
                        nature_draw_list.append(nature_draw)
                judge_value = len(nature_set) * 100 / 2
                nature_type_text = ""
                if type_value >= judge_value:
                    nature_type_text = type_config.good
                else:
                    nature_type_text = type_config.bad
                nature_draw = draw.LittleTitleLineDraw(nature_type_text, self.width, ":")
                self.draw_list.append(nature_draw)
                self.draw_list.append(nature_draw_list)
            for value in self.draw_list:
                if isinstance(value, list):
                    now_draw = panel.VerticalDrawTextListGroup(self.width)
                    now_group = value_handle.list_of_groups(value, 1)
                    now_draw.draw_list = now_group
                    now_draw.draw()
                else:
                    value.draw()
            now_ask_list = [_("保存"), _("取消")]
            askfor_panel = panel.OneMessageAndSingleColumnButton()
            askfor_panel.set(now_ask_list,"",0)
            askfor_panel.draw()
            now_ask_return_list = list(askfor_panel.get_return_list().keys())
            ask_list.extend(now_ask_return_list)
            yrn = flow_handle.askfor_all(ask_list)
            if yrn == now_ask_return_list[0]:
                cache.character_data[0].nature = self.nature
                break
            elif yrn == now_ask_return_list[1]:
                break

    def change_nature(self, nature_id: int):
        if self.nature[nature_id] >= 50:
            self.nature[nature_id] -= 50
        else:
            self.nature[nature_id] += 50

