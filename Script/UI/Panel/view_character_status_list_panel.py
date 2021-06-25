import datetime
from types import FunctionType
from Script.UI.Moudle import panel, draw
from Script.Core import get_text, game_type, cache_control, text_handle, py_cmd, constant, flow_handle
from Script.Config import game_config, normal_config
from Script.Design import character_move, attr_text, game_time

_: FunctionType = get_text._
""" 翻译api """
cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
line_feed = draw.NormalDraw()
""" 换行绘制对象 """
line_feed.text = "\n"
line_feed.width = 1
window_width: int = normal_config.config_normal.text_width
""" 窗体宽度 """


class CharacterStatusListPanel:
    """
    用于查看角色状态监控界面面板对象
    Keyword arguments:
    width -- 绘制宽度
    """

    def __init__(self, width: int):
        """初始化绘制对象"""
        self.width: int = width
        self.handle_panel: panel.PageHandlePanel = None
        """ 当前角色列表控制面板 """
        self.now_panel = _("存活")
        """ 当前面板类型 """
        self.live_list = [
            character_id
            for character_id in cache.character_data
            if cache.character_data[character_id].state != constant.CharacterStatus.STATUS_DEAD
        ]
        """ 存活的角色列表 """
        self.dead_list = [
            character_id
            for character_id in cache.character_data
            if cache.character_data[character_id].state == constant.CharacterStatus.STATUS_DEAD
        ]
        """ 死亡的角色列表 """

    def draw(self):
        """绘制对象"""
        title_draw = draw.TitleLineDraw(_("Alpha监控台"), self.width)
        panel_type_list = [_("存活"), _("死亡")]
        panel_data = [
            panel.PageHandlePanel(self.live_list, SeeCharacterStatusDraw, 10, 1, window_width, 1, 1, 0),
            panel.PageHandlePanel(self.dead_list, SeeCharacterStatusDraw, 10, 1, window_width, 1, 1, 0),
        ]
        while 1:
            if cache.now_panel_id != constant.Panel.VIEW_CHARACTER_STATUS_LIST:
                break
            if self.now_panel == panel_type_list[0]:
                self.handle_panel = panel_data[0]
            else:
                self.handle_panel = panel_data[1]
            self.handle_panel.update()
            title_draw.draw()
            return_list = []
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
            line_feed.draw()
            line = draw.LineDraw("+", self.width)
            line.draw()
            self.handle_panel.draw()
            return_list.extend(self.handle_panel.return_list)
            back_draw = draw.CenterButton(_("[返回]"), _("返回"), window_width)
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
    点击后可移动至该角色位置的角色状态按钮对象
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
        index_text = text_handle.id_index(button_id)
        status_text = game_config.config_status[character_data.state].name
        position_text = attr_text.get_scene_path_text(character_data.position)
        if character_data.dead:
            cause_of_death_config = game_config.config_cause_of_death[character_data.cause_of_death]
            death_time_text = _("死亡时间:")
            death_time = datetime.datetime.fromtimestamp(
                character_data.behavior.start_time, game_time.time_zone
            )
            death_time_text = f"{death_time_text}{str(death_time)}"
            button_text = f"{index_text} {character_data.name} {cause_of_death_config.name} {death_time_text} {position_text}"
        else:
            button_text = f"{index_text} {character_data.name} {status_text} {position_text}"
        now_draw = draw.LeftButton(
            button_text, self.button_return, self.width, cmd_func=self.move_to_character
        )
        self.now_draw = now_draw
        """ 绘制的对象 """

    def draw(self):
        """绘制对象"""
        self.now_draw.draw()

    def move_to_character(self):
        """移动至角色所在场景"""
        character_data: game_type.Character = cache.character_data[self.text]
        py_cmd.clr_cmd()
        line_feed.draw()
        cache.wframe_mouse.w_frame_skip_wait_mouse = 1
        character_move.own_charcter_move(character_data.position)
