from typing import Dict, Tuple, List
from types import FunctionType
from Script.UI.Moudle import draw,panel
from Script.Core import cache_contorl, get_text
from Script.Config import game_config
from Script.Design import attr_text,map_handle,attr_calculation

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
        self.max_width = width
        """ 绘制的最大宽度 """
        self.now_panel = _("属性")
        """ 当前的属性页id """
        self.character_id = character_id
        """ 要绘制的角色id """
        main_attr_draw = SeeCharacterMainAttrPanel(character_id,width)
        self.draw_data = {
            _("属性"):main_attr_draw,
            _("状态"):None,
            _("服装"):None,
            _("道具"):None,
            _("穿戴"):None,
            _("经验"):None,
            _("技能"):None,
            _("语言"):None,
            _("性格"):None,
            _("社交"):None
        }
        """ 按钮文本对应属性面板 """

    def draw(self):
        """ 绘制面板 """
        self.draw_data[self.now_panel].draw()

class SeeCharacterMainAttrPanel:
    """ 显示角色主属性面板对象 """

    def __init__(self,character_id:int,width:int):
        """
        初始化绘制对象
        Keyword arguments:
        character_id -- 角色id
        width -- 绘制宽度
        """
        head_draw = CharacterInfoHead(character_id, width)
        stature_draw = CharacterStatureText(character_id,width)
        room_draw = CharacterRoomText(character_id,width)
        birthday_draw = CharacterBirthdayText(character_id,width)
        sture_info_draw = CharacterStatureInfoText(character_id,width)
        measurement_draw = CharacterMeasurementsText(character_id,width)
        self.draw_list: List[draw.NormalDraw] = [
            head_draw,
            stature_draw,
            room_draw,
            birthday_draw,
            sture_info_draw,
            measurement_draw
        ]
        """ 绘制的面板列表 """

    def draw(self):
        """ 绘制面板 """
        for label in self.draw_list:
            label.draw()


class SeeCharacterStatusPanel:
    """ 显示角色状态面板对象 """

    def __init__(self,character_id:int,width:int):
        """
        初始化绘制对象
        Keyword arguments:
        character_id -- 角色id
        width -- 绘制宽度
        """
        head_draw = CharacterInfoHead(character_id,width)
        self.draw_list: List[draw.NormalDraw] = [
            head_draw
        ]
        """ 绘制的面板列表 """

    def draw(self):
        """ 绘制面板 """
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
        dormitory_text = ""
        if dormitory == "":
            dormitory_text = _("暂无")
        else:
            dormitory_path = map_handle.get_map_system_path_for_str(dormitory)
            dormitory_text = attr_text.get_scene_path_text(dormitory_path)
        self.dormitory_text = _(f"宿舍位置:{dormitory_text}")
        """ 宿舍位置文本 """
        classroom = character_data.classroom
        now_classroom_text = _("教室位置:")
        if classroom != "":
            classroom_path = map_handle.get_map_system_path_for_str(classroom)
            classroom_text = attr_text.get_scene_path_text(classroom_path)
            now_classroom_text += classroom_text
        else:
            now_classroom_text += _("暂无")
        self.classroom_text = now_classroom_text
        """ 教室位置文本 """
        officeroom = character_data.officeroom
        now_officeroom_text = _("办公室位置:")
        if len(officeroom):
            officeroom_text = attr_text.get_scene_path_text(officeroom)
            now_officeroom_text += officeroom_text
        else:
            now_officeroom_text += _("暂无")
        self.officeroom_text = now_officeroom_text
        """ 办公室位置文本 """

    def draw(self):
        """ 绘制面板 """
        line = draw.LineDraw(".",self.max_width)
        line.draw()
        info_draw = panel.CenterDrawTextListPanel()
        info_draw.set([self.dormitory_text,self.classroom_text,self.officeroom_text],self.max_width,3)
        info_draw.draw()


class CharacterBirthdayText:
    """ 角色年龄/生日信息显示面板 """

    def __init__(self,character_id:int,width:int):
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
        age_text = _(f"年龄:{character_data.age}岁")
        birthday_text = _(f"生日:{character_data.birthday.month}月{character_data.birthday.day}日")
        self.info_list = [age_text,birthday_text]
        """ 绘制的文本列表 """

    def draw(self):
        """ 绘制面板 """
        line = draw.LineDraw(".",self.max_width)
        line.draw()
        info_draw = panel.CenterDrawTextListPanel()
        info_draw.set(self.info_list,self.max_width,2)
        info_draw.draw()

class CharacterStatureInfoText:
    """ 角色身高体重罩杯信息显示面板 """

    def __init__(self,character_id:int,width:int):
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
        now_height = round(character_data.height.now_height,2)
        now_height_text = _(f"身高:{now_height}")
        now_weight = round(character_data.weight,2)
        now_weight_text = _(f"体重:{now_weight}")
        now_chest_tem_id = attr_calculation.judge_chest_group(character_data.chest.now_chest)
        now_chest_tem = game_config.config_chest[now_chest_tem_id]
        now_chest_text = _(f"罩杯:{now_chest_tem.info}")
        self.info_list = [now_height_text,now_chest_text,now_weight_text]
        """ 绘制的文本列表 """

    def draw(self):
        """ 绘制面板 """
        line = draw.LineDraw(".",self.max_width)
        line.draw()
        info_draw = panel.CenterDrawTextListPanel()
        info_draw.set(self.info_list,self.max_width,3)
        info_draw.draw()


class CharacterMeasurementsText:
    """ 角色三围信息显示面板 """

    def __init__(self,character_id:int,width:int):
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
        character_data.measurements.bust
        character_data.measurements.hip
        character_data.measurements.waist
        now_bust = round(character_data.measurements.bust,2)
        now_hip = round(character_data.measurements.hip,2)
        now_waist = round(character_data.measurements.waist,2)
        now_bust_text = _(f"胸围:{now_bust}")
        now_waist_text = _(f"腰围:{now_waist}")
        now_hip_text = _(f"臀围:{now_hip}")
        self.info_list = [now_bust_text,now_waist_text,now_hip_text]
        """ 绘制的文本列表 """

    def draw(self):
        """ 绘制面板 """
        line = draw.LineDraw(".",self.max_width)
        line.draw()
        info_draw = panel.CenterDrawTextListPanel()
        info_draw.set(self.info_list,self.max_width,3)
        info_draw.draw()
