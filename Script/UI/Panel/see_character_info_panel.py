from typing import Dict, Tuple, List
from types import FunctionType
from Script.UI.Moudle import draw
from Script.Core import cache_contorl, get_text
from Script.Config import game_config
from Script.Design import attr_text,map_handle

panel_info_data = {}

_: FunctionType = get_text._
""" 翻译api """


class SeeCharacterInfoPanel:
    """ 用于查看角色属性的面板对象 """

    def __init__(self, character_id: int, width: int):
        """
        初始化绘制对象
        Keyword arguments:
        character_id -- 角色id
        width -- 绘制宽度
        """
        self.button_list: List[draw.Button] = []
        """ 控制切换属性页的按钮列表 """
        self.return_list: Dict[str, str] = {}
        """
        切换属性页按钮的响应列表
        按钮返回值:按钮文本
        """
        self.max_width = width
        """ 绘制的最大宽度 """
        self.now_panel = 0
        """ 当前的属性页id """
        self.character_id = character_id
        """ 要绘制的角色id """
        head_draw = CharacterInfoHead(character_id, width)
        stature_draw = CharacterStatureText(character_id,width)
        room_draw = CharacterRoomText(character_id,width)
        self.draw_list: List[draw.NormalDraw] = [head_draw,stature_draw,room_draw]
        """ 绘制的面板列表 """

    def draw(self):
        for label in self.draw_list:
            label.draw()


class CharacterInfoHead:
    """ 角色信息面板头部面板 """

    def __init__(self, character_id: int, width: int):
        """
        初始化绘制对象
        Keyword arguments:
        character_id -- 角色id
        width -- 最大宽度
        """
        self.character_id = character_id
        """ 要绘制的角色id """
        self.max_width = width
        """ 当前最大可绘制宽度 """
        character_data = cache_contorl.character_data[character_id]
        message = _(f"No.{character_id} 姓名:{character_data.name} 称呼:{character_data.nick_name}")
        message_draw = draw.CenterDraw()
        message_draw.max_width = width / 2
        message_draw.text = message
        hp_draw = draw.InfoBarDraw()
        hp_draw.max_width = width / 2
        hp_draw.set(
            "HitPointbar",
            character_data.hit_point_max,
            character_data.hit_point,
            _("体力"),
        )
        mp_draw = draw.InfoBarDraw()
        mp_draw.max_width = width / 2
        mp_draw.set(
            "ManaPointbar",
            character_data.mana_point_max,
            character_data.mana_point,
            _("气力"),
        )
        status_text = game_config.config_status[character_data.state].name
        status_draw = draw.CenterDraw()
        status_draw.max_width = width / 2
        status_draw.text = _(f"状态:{status_text}")
        self.draw_list: List[Tuple[draw.NormalDraw, draw.NormalDraw]] = [
            (message_draw, hp_draw),
            (status_draw, mp_draw),
        ]
        """ 要绘制的面板列表 """

    def draw(self):
        """ 绘制面板 """
        line_feed = draw.NormalDraw()
        line_feed.text = "\n"
        line_feed.max_width = 1
        title_draw = draw.TitleLineDraw(_("人物属性"), self.max_width)
        title_draw.draw()
        for draw_tuple in self.draw_list:
            for label in draw_tuple:
                label.draw()
            line_feed.draw()


class CharacterStatureText:
    """ 身材描述信息面板 """

    def __init__(self, character_id: int, width: int):
        """
        初始化绘制对象
        Keyword arguments:
        character_id -- 角色id
        width -- 最大宽度
        """
        self.character_id = character_id
        """ 要绘制的角色id """
        self.max_width = width
        """ 当前最大可绘制宽度 """
        player_data = cache_contorl.character_data[0]
        description = attr_text.get_stature_text(character_id)
        description = description.format(
            Name=player_data.name,
            NickName=player_data.nick_name,
        )
        self.description = description
        """ 身材描述文本 """

    def draw(self):
        """ 绘制面板 """
        line = draw.LineDraw(":",self.max_width)
        line.draw()
        info_draw = draw.CenterDraw()
        info_draw.text = self.description
        info_draw.max_width = self.max_width
        info_draw.draw()
        line_feed = draw.NormalDraw()
        line_feed.text = "\n"
        line_feed.max_width = 1
        line_feed.draw()

class CharacterRoomText:
    """ 角色宿舍/教室和办公室地址显示面板 """

    def __init__(self,character_id: int,width: int):
        """
        初始化绘制对象
        Keyword arguments:
        character_id -- 角色id
        width -- 最大宽度
        """
        self.character_id = character_id
        """ 要绘制的角色id """
        self.max_width = width
        """ 当前最大可绘制宽度 """
        character_data = cache_contorl.character_data[self.character_id]
        dormitory = character_data.dormitory
        dormitory_path = map_handle.get_map_system_path_for_str(dormitory)
        dormitory_text = attr_text.get_scene_path_text(dormitory_path)
        dormitory_draw = draw.CenterDraw()
        dormitory_draw.max_width = width / 3
        dormitory_draw.text = _(f"宿舍位置:{dormitory_text}")
        classroom = character_data.classroom
        now_classroom_text = _("教室位置:")
        if classroom != "":
            classroom_path = map_handle.get_map_system_path_for_str(classroom)
            classroom_text = attr_text.get_scene_path_text(classroom_path)
            now_classroom_text += classroom_text
        else:
            now_classroom_text += _("暂无")
        classroom_draw = draw.CenterDraw()
        classroom_draw.max_width = width / 3
        classroom_draw.text = now_classroom_text
        officeroom = character_data.officeroom
        now_officeroom_text = _("办公室位置:")
        if len(officeroom):
            officeroom_text = attr_text.get_scene_path_text(officeroom)
            now_officeroom_text += officeroom_text
        else:
            now_officeroom_text += _("暂无")
        officeroom_draw = draw.CenterDraw()
        officeroom_draw.max_width = width / 3
        officeroom_draw.text = now_officeroom_text
        self.draw_list:List[draw.NormalDraw] = [dormitory_draw,classroom_draw,officeroom_draw]
        """ 绘制的文本列表 """

    def draw(self):
        """ 绘制面板 """
        line = draw.LineDraw(".",self.max_width)
        line.draw()
        for now_draw in self.draw_list:
            now_draw.draw()
        line_feed = draw.NormalDraw()
        line_feed.text = "\n"
        line_feed.max_width = 1
        line_feed.draw()
