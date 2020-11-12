from uuid import UUID
from typing import Dict, Tuple, List
from types import FunctionType
from Script.UI.Moudle import draw,panel
from Script.Core import cache_contorl, get_text, value_handle, game_type, text_handle
from Script.Config import game_config
from Script.Design import attr_text,map_handle,attr_calculation

panel_info_data = {}

_: FunctionType = get_text._
""" 翻译api """

line_feed = draw.NormalDraw()
""" 换行绘制对象 """
line_feed.text = "\n"
line_feed.max_width = 1


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
        see_status_draw = SeeCharacterStatusPanel(character_id,width,5)
        see_clothing_draw = SeeCharacterPutOnClothingListPanel(character_id,width)
        self.draw_data = {
            _("属性"):main_attr_draw,
            _("状态"):see_status_draw,
            _("服装"):see_clothing_draw,
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


class SeeCharacterPutOnClothingListPanel:
    """ 显示角色已穿戴服装面板 """

    def __init__(self,character_id:int,width:int):
        """
        初始化绘制对象
        Keyword arguments:
        character_id -- 角色id
        width -- 绘制宽度
        """
        self.character_id:int = character_id
        """ 绘制的角色id """
        self.width:int = width
        """ 最大绘制宽度 """

    def draw(self):
        """ 绘制面板 """
        character_data = cache_contorl.character_data[self.character_id]
        title_draw = draw.TitleLineDraw(_("人物服装"), self.width)
        title_draw.draw()
        draw_list = []
        id_width = 0
        for clothing_type in game_config.config_clothing_type:
            type_data = game_config.config_clothing_type[clothing_type]
            type_draw = draw.LittleTitleLineDraw(type_data.name,self.width,":")
            draw_list.append(type_draw)
            if clothing_type in character_data.put_on and isinstance(character_data.put_on[clothing_type],UUID):
                now_draw = ClothingInfoDrawPanel(self.character_id,clothing_type,character_data.put_on[clothing_type],self.width)
                now_id_width = text_handle.get_text_index(now_draw.text_list[0])
                if now_id_width > id_width:
                    id_width = now_id_width
            else:
                now_draw = draw.NormalDraw()
                now_draw.text = _("未穿戴")
                now_draw.max_width = self.width
            draw_list.append(now_draw)
            draw_list.append(line_feed)
        for value in draw_list:
            if "id_width" in value.__dict__:
                value.id_width = id_width
            value.draw()


class ClothingInfoDrawPanel:
    """ 服装信息绘制面板 """

    def __init__(self,character_id:int,clothing_type:int,clothing_id:UUID,width:int,draw_button:bool=False,button_id:int=0):
        """
        初始化绘制对象
        Keyword arguments:
        character_id -- 角色id
        clothing_type -- 服装类型
        clothing_type -- 服装id
        width -- 绘制宽度
        draw_button -- 是否按按钮绘制
        button_id -- 绘制按钮时的id
        """
        character_data = cache_contorl.character_data[character_id]
        self.clothing_data:game_type.Clothing = character_data.clothing[clothing_type][clothing_id]
        """ 当前服装数据 """
        self.width:int = width
        """ 最大绘制宽度 """
        self.draw_button:bool = draw_button
        """ 是否按按钮绘制 """
        self.button_id:int = button_id
        """ 绘制按钮时的id """
        self.id_width:int = 0
        """ 绘制时计算用的id宽度 """
        now_id_text = ""
        if self.draw_button:
            now_id_text = text_handle.id_index(self.button_id)
        fix_width = self.width - len(now_id_text)
        value_dict = {
            _("可爱"):self.clothing_data.sweet,
            _("性感"):self.clothing_data.sexy,
            _("帅气"):self.clothing_data.handsome,
            _("清新"):self.clothing_data.fresh,
            _("典雅"):self.clothing_data.elegant,
            _("清洁"):self.clothing_data.cleanliness,
            _("保暖"):self.clothing_data.warm,
        }
        describe_list = [
            _("可爱的"),
            _("性感的"),
            _("帅气的"),
            _("清新的"),
            _("典雅的"),
            _("清洁的"),
            _("保暖的")
        ]
        value_list = list(value_dict.values())
        describe_id = value_list.index(max(value_list))
        describe = describe_list[describe_id]
        clothing_config = game_config.config_clothing_tem[self.clothing_data.tem_id]
        clothing_name = f"{self.clothing_data.evaluation}{describe}{clothing_config.name}"
        fix_width -= text_handle.get_text_index(clothing_name)
        value_text = ""
        for value_id in value_dict:
            value = str(value_dict[value_id])
            if len(value) < 4:
                value = (4 - len(value)) * " " + value
            value_text += f"|{value_id}:{value}"
        value_text += "|"
        id_text = ""
        if self.draw_button:
            id_text = f"{now_id_text} {clothing_name}"
        else:
            id_text = clothing_name
        self.text_list:List[str] = [id_text,value_text]
        """ 绘制的文本列表 """

    def draw(self):
        self.text_list[1] = text_handle.align(self.text_list[1],"center",0,1,self.width-self.id_width)
        text_width = text_handle.get_text_index(self.text_list[0])
        if text_width < self.id_width:
            self.text_list[0] += " " * (self.id_width - text_width)
        now_text = f"{self.text_list[0]}{self.text_list[1]}"
        if self.draw_button:
            now_draw = draw.Button(now_text,str(self.button_id))
        else:
            now_draw = draw.NormalDraw()
            now_draw.text = now_text
        now_draw.max_width = self.width
        now_draw.draw()


class SeeCharacterStatusPanel:
    """ 显示角色状态面板对象 """

    def __init__(self,character_id:int,width:int,column:int):
        """
        初始化绘制对象
        Keyword arguments:
        character_id -- 角色id
        width -- 绘制宽度
        column -- 每行状态最大个数
        """
        self.character_id = character_id
        """ 要绘制的角色id """
        self.width = width
        """ 面板最大宽度 """
        self.column = column
        """ 每行状态最大个数 """
        self.draw_list:List[draw.NormalDraw] = []
        """ 绘制的文本列表 """
        character_data = cache_contorl.character_data[character_id]
        for status_type in game_config.config_character_state_type_data:
            type_data = game_config.config_character_state_type[status_type]
            type_line = draw.LittleTitleLineDraw(type_data.name,width,":")
            self.draw_list.append(type_line)
            type_set = game_config.config_character_state_type_data[status_type]
            status_text_list = []
            for status_id in type_set:
                if status_type == 0:
                    if character_data.sex == 0:
                        if status_id in {2,3,6}:
                            continue
                    elif character_data.sex == 1:
                        if status_id == 5:
                            continue
                    elif character_data.sex == 3:
                        if status_id in {2,3,5,6}:
                            continue
                status_text = game_config.config_character_state[status_id].name
                status_value = 0
                if status_id in character_data.status:
                    status_value = character_data.status[status_id]
                status_value = round(status_value)
                now_text = f"{status_text}:{status_value}"
                status_text_list.append(now_text)
            now_draw = panel.CenterDrawTextListPanel()
            now_draw.set(status_text_list,self.width,self.column)
            self.draw_list.extend(now_draw.draw_list)

    def draw(self):
        """ 绘制面板 """
        title_draw = draw.TitleLineDraw(_("人物状态"), self.width)
        title_draw.draw()
        for label in self.draw_list:
            if isinstance(label,list):
                for value in label:
                    value.draw()
                line_feed.draw()
            else:
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
        sex_text = game_config.config_sex_tem[character_data.sex].name
        message = _(f"No.{character_id} 姓名:{character_data.name} 称呼:{character_data.nick_name} 性别:{sex_text}")
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
