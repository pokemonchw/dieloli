from typing import List, Dict
from types import FunctionType
from Script.UI.Moudle import draw, panel
from Script.Core import (
    get_text,
    cache_control,
    game_type,
    flow_handle,
    value_handle,
    text_handle,
    py_cmd,
)
from Script.Design import map_handle, attr_text, character_move, constant
from Script.Config import game_config

_: FunctionType = get_text._
""" 翻译api """
cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
line_feed = draw.NormalDraw()
""" 换行绘制对象 """
line_feed.text = "\n"
line_feed.width = 1


class SeeMapPanel:
    """
    用于查看当前地图界面面板对象
    Keyword arguments:
    width -- 绘制宽度
    """

    def __init__(self, width: int):
        """初始化绘制对象"""
        self.width: int = width
        """ 绘制的最大宽度 """
        character_data: game_type.Character = cache.character_data[0]
        self.now_map: List[str] = map_handle.get_map_for_path(character_data.position)
        """ 当前查看的地图坐标 """

    def draw(self):
        """绘制对象"""
        move_menu_panel_data = {
            0: MapSceneNameDraw(self.now_map, self.width),
            1: GlobalSceneNamePanel(self.now_map, self.width),
            2: SocialSceneNamePanel(self.now_map, self.width),
            3: CollectionSceneNamePanel(self.now_map, self.width),
        }
        move_menu_panel = MoveMenuPanel(self.width)
        while 1:
            line_feed.draw()
            if cache.now_panel_id != constant.Panel.SEE_MAP:
                break
            map_path_str = map_handle.get_map_system_path_str_for_list(self.now_map)
            map_data: game_type.Map = cache.map_data[map_path_str]
            map_name = attr_text.get_map_path_text(self.now_map)
            title_draw = draw.TitleLineDraw(_("移动:") + map_name, self.width)
            title_draw.draw()
            now_draw_list: game_type.MapDraw = map_data.map_draw
            character_data: game_type.Character = cache.character_data[0]
            character_scene_id = map_handle.get_map_scene_id_for_scene_path(
                self.now_map, character_data.position
            )
            return_list = []
            index = 0
            for now_draw_line in now_draw_list.draw_text:
                now_draw_line: game_type.MapDrawLine = now_draw_line
                fix_width = int((self.width - now_draw_line.width) / 2)
                fix_text = " " * fix_width
                fix_draw = draw.NormalDraw()
                fix_draw.text = fix_text
                fix_draw.width = fix_width
                fix_draw.draw()
                for draw_text in now_draw_line.draw_list:
                    if draw_text.is_button and draw_text.text != character_scene_id:
                        scene_path = map_handle.get_scene_path_for_map_scene_id(
                            self.now_map, draw_text.text
                        )
                        now_draw = draw.Button(
                            draw_text.text,
                            draw_text.text,
                            cmd_func=self.move_now,
                            args=(scene_path,),
                        )
                        now_draw.width = self.width
                        now_draw.draw()
                        return_list.append(now_draw.return_text)
                    else:
                        now_draw = draw.NormalDraw()
                        now_draw.text = draw_text.text
                        now_draw.width = self.width
                        if draw_text.is_button and draw_text.text == character_scene_id:
                            now_draw.style = "nowmap"
                        now_draw.draw()
                line_feed.draw()
            path_edge = map_data.path_edge
            scene_path = path_edge[character_scene_id].copy()
            if character_scene_id in scene_path:
                del scene_path[character_scene_id]
            scene_path_list = list(scene_path.keys())
            if scene_path_list:
                line = draw.LineDraw(".", self.width)
                line.draw()
                message_draw = draw.NormalDraw()
                message_draw.text = _("你可以从这里前往:\n")
                message_draw.width = self.width
                message_draw.draw()
                draw_list = []
                character_scene_str = map_handle.get_map_system_path_str_for_list(
                    character_data.position
                )
                for scene in scene_path_list:
                    load_scene_data = map_handle.get_scene_data_for_map(map_path_str, scene)
                    now_scene_path = map_handle.get_map_system_path_for_str(
                        load_scene_data.scene_path
                    )
                    now_move_time = map_handle.scene_move_time[character_scene_str][
                        load_scene_data.scene_path
                    ]
                    now_draw = draw.CenterButton(
                        _("[{scene_name}:{move_time}分钟]").format(
                            scene_name=load_scene_data.scene_name, move_time=now_move_time
                        ),
                        load_scene_data.scene_name,
                        self.width / 4,
                        cmd_func=self.move_now,
                        args=(now_scene_path,),
                    )
                    return_list.append(now_draw.return_text)
                    draw_list.append(now_draw)
                draw_group = value_handle.list_of_groups(draw_list, 4)
                for now_draw_list in draw_group:
                    for now_draw in now_draw_list:
                        now_draw.draw()
                    line_feed.draw()
            scene_id_list = list(path_edge.keys())
            now_index = len(scene_id_list)
            index = now_index
            move_menu_panel.update()
            move_menu_panel.draw()
            return_list.extend(move_menu_panel.return_list)
            if move_menu_panel.now_type in move_menu_panel_data:
                now_move_menu = move_menu_panel_data[move_menu_panel.now_type]
                now_move_menu.update(self.now_map, index)
                now_move_menu.draw()
                now_index = now_move_menu.end_index + 1
                return_list.extend(now_move_menu.return_list)
            line = draw.LineDraw("=", self.width)
            line.draw()
            if self.now_map != []:
                now_id = text_handle.id_index(now_index)
                now_text = now_id + _("查看上级地图")
                up_button = draw.CenterButton(
                    now_text, str(now_index), self.width / 3, cmd_func=self.up_map
                )
                up_button.draw()
                return_list.append(up_button.return_text)
                now_index += 1
            else:
                now_draw = draw.NormalDraw()
                now_draw.text = " " * int(self.width / 3)
                now_draw.width = self.width / 3
                now_draw.draw()
            back_id = text_handle.id_index(now_index)
            now_text = back_id + _("返回")
            back_button = draw.CenterButton(now_text, str(now_index), self.width / 3)
            back_button.draw()
            return_list.append(back_button.return_text)
            now_index += 1
            character_map = map_handle.get_map_for_path(character_data.position)
            if character_map != self.now_map:
                now_id = text_handle.id_index(now_index)
                now_text = now_id + _("查看下级地图")
                down_button = draw.CenterButton(
                    now_text, str(now_index), self.width / 3, cmd_func=self.down_map
                )
                down_button.draw()
                return_list.append(down_button.return_text)
            line_feed.draw()
            yrn = flow_handle.askfor_all(return_list)
            py_cmd.clr_cmd()
            if yrn == back_button.return_text:
                cache.now_panel_id = constant.Panel.IN_SCENE
                break

    def up_map(self):
        """将当前地图切换为上级地图"""
        py_cmd.clr_cmd()
        up_map_path = map_handle.get_map_for_path(self.now_map)
        self.now_map = up_map_path

    def down_map(self):
        """将当前地图切换为下级地图"""
        py_cmd.clr_cmd()
        character_position = cache.character_data[0].position
        down_map_scene_id = map_handle.get_map_scene_id_for_scene_path(
            self.now_map, character_position
        )
        self.now_map.append(down_map_scene_id)

    @staticmethod
    def move_now(scene_path: List[str]):
        """
        控制角色移动至指定场景
        Keyword arguments:
        scene_path -- 目标场景路径
        """
        py_cmd.clr_cmd()
        line_feed.draw()
        cache.wframe_mouse.w_frame_skip_wait_mouse = 1
        character_move.own_charcter_move(scene_path)


class MoveMenuPanel:
    """
    快捷移动菜单面板
    Keyword arguments:
    width -- 绘制宽度
    """

    def __init__(self, width: int):
        """初始化绘制对象"""
        self.width: int = width
        """ 最大绘制宽度 """
        self.return_list: List[str] = []
        """ 监听的按钮列表 """
        self.now_type: int = 0
        """ 当前的移动菜单类型 """
        self.draw_list: List[draw.NormalDraw] = []
        """ 绘制的对象列表 """
        self.move_type_id_data: Dict[str, int] = {
            game_config.config_move_menu_type[i].name: i for i in game_config.config_move_menu_type
        }
        """ 移动类型名字对应配表id """

    def update(self):
        """更新绘制面板"""
        line = draw.LineDraw(".", self.width)
        self.draw_list = []
        self.return_list = []
        self.draw_list.append(line)
        menu_draw = panel.CenterDrawButtonListPanel()
        move_name_list = [
            game_config.config_move_menu_type[i].name for i in game_config.config_move_menu_type
        ]
        move_name_draw_list = [f"[{name}]" for name in move_name_list]
        menu_draw.set(
            move_name_draw_list,
            move_name_list,
            self.width,
            len(game_config.config_move_menu_type),
            move_name_draw_list[self.now_type],
            self.change_type,
        )
        self.draw_list.append(menu_draw)
        self.return_list.extend(menu_draw.return_list)

    def change_type(self, to_type: str):
        """
        改变当前快捷移动菜单类型
        Keyword arguments:
        to_type -- 指定的新类型id
        """
        self.now_type = self.move_type_id_data[to_type]
        py_cmd.clr_cmd()

    def draw(self):
        """绘制面板"""
        for now_draw in self.draw_list:
            now_draw.draw()
        line = draw.LineDraw("-.-", self.width)
        line.draw()


class MapSceneNameDraw:
    """
    绘制指定地图地图场景id对应场景名列表
    Keyword arguments:
    now_map -- 地图路径
    width -- 绘制宽度
    """

    def __init__(self, now_map: List[str], width: int):
        self.width: int = width
        """ 绘制的最大宽度 """
        self.now_map: List[str] = now_map
        """ 当前查看的地图坐标 """
        self.return_list: List[str] = []
        """ 当前面板的按钮返回 """
        self.end_index: int = 0
        """ 结束按钮id """

    def update(self, now_map: List[str], _):
        """
        更新当前面板对象
        Keyword arguments:
        now_map -- 当前地图
        """
        self.now_map = now_map

    def draw(self):
        """绘制面板"""
        self.return_list = []
        map_path_str = map_handle.get_map_system_path_str_for_list(self.now_map)
        map_data: game_type.Map = cache.map_data[map_path_str]
        path_edge = map_data.path_edge
        scene_id_list = list(path_edge.keys())
        if scene_id_list:
            character_data: game_type.Character = cache.character_data[0]
            character_scene_id = map_handle.get_map_scene_id_for_scene_path(
                self.now_map, character_data.position
            )
            character_scene_str = map_handle.get_map_system_path_str_for_list(
                character_data.position
            )
            scene_path = path_edge[character_scene_id].copy()
            if character_scene_id in scene_path:
                del scene_path[character_scene_id]
            draw_list = []
            for scene_id in scene_id_list:
                load_scene_data = map_handle.get_scene_data_for_map(map_path_str, scene_id)
                now_scene_path = map_handle.get_map_system_path_for_str(load_scene_data.scene_path)
                path_str = load_scene_data.scene_path
                move_time = 0
                if path_str in map_handle.scene_move_time[character_scene_str]:
                    move_time = map_handle.scene_move_time[character_scene_str][path_str]
                now_id_text = _("{scene_id}:{scene_name}:{move_time}分钟").format(
                    scene_id=scene_id, scene_name=load_scene_data.scene_name, move_time=move_time
                )
                now_draw = draw.LeftButton(
                    now_id_text,
                    now_id_text,
                    self.width,
                    cmd_func=self.move_now,
                    args=(now_scene_path,),
                )
                self.return_list.append(now_draw.return_text)
                draw_list.append(now_draw)
            draw_group = value_handle.list_of_groups(draw_list, 4)
            now_width_index = 0
            for now_draw_list in draw_group:
                if len(now_draw_list) > now_width_index:
                    now_width_index = len(now_draw_list)
            now_width = self.width / now_width_index
            for now_draw_list in draw_group:
                for now_draw in now_draw_list:
                    now_draw.width = now_width
                    now_draw.draw()
                line_feed.draw()
        self.end_index = len(scene_id_list) - 1

    @staticmethod
    def move_now(scene_path: List[str]):
        """
        控制角色移动至指定场景
        Keyword arguments:
        scene_path -- 目标场景路径
        """
        py_cmd.clr_cmd()
        line_feed.draw()
        cache.wframe_mouse.w_frame_skip_wait_mouse = 1
        character_move.own_charcter_move(scene_path)


class GlobalSceneNamePanel:
    """
    绘制公共快捷寻路场景按钮列表
    Keyword arguments:
    now_map -- 地图路径
    width -- 绘制宽度
    """

    def __init__(self, now_map: List[str], width: int):
        self.width: int = width
        """ 绘制的最大宽度 """
        self.now_map: List[str] = now_map
        """ 当前查看的地图坐标 """
        self.return_list: List[str] = []
        """ 当前面板的按钮返回 """
        self.end_index: int = 0
        """ 结束按钮id """
        self.handle_panel = panel.PageHandlePanel(
            self.get_path_list(), ScenePathNameMoveDraw, 20, 3, self.width, 1
        )
        self.end_index = self.handle_panel.end_index

    @staticmethod
    def get_path_list() -> list:
        """
        获取公共场景路径列表
        Return arguments:
        list -- 路径列表
        """
        character_data: game_type.Character = cache.character_data[0]
        classroom_path = []
        officeroom_path = []
        if 0 in character_data.identity_data:
            identity_data: game_type.StudentIdentity = character_data.identity_data[0]
            classroom_path = map_handle.get_map_system_path_for_str(identity_data.classroom)
        elif 1 in character_data.identity_data:
            identity_data: game_type.TeacherIdentity = character_data.identity_data[1]
            if identity_data.now_classroom != "":
                classroom_path = map_handle.get_map_system_path_for_str(identity_data.now_classroom)
            officeroom_path = map_handle.get_map_system_path_for_str(identity_data.officeroom)
        square_path = ["2"]
        swim_path = ["14", "0"]
        big_restaurant_path = ["10", "0", "0"]
        small_restaurant_path = ["16", "0", "0"]
        multi_media_class_room_a_path = ["1", "3", "1"]
        multi_media_class_room_b_path = ["1", "3", "2"]
        music_class_room_path = ["1", "4", "1"]
        student_union_office_path = ["3", "1", "4"]
        computer_room_a_path = ["3", "3", "1"]
        computer_room_b_path = ["3", "3", "2"]
        shop_path = ["11"]
        dormitory_path = map_handle.get_map_system_path_for_str(character_data.dormitory)
        path_list = [
            dormitory_path,
            classroom_path,
            officeroom_path,
            square_path,
            swim_path,
            big_restaurant_path,
            small_restaurant_path,
            multi_media_class_room_a_path,
            multi_media_class_room_b_path,
            computer_room_a_path,
            computer_room_b_path,
            music_class_room_path,
            shop_path,
            student_union_office_path,
        ]
        path_list = [i for i in path_list if len(i)]
        return path_list

    def update(self, now_map: List[str], start_index: int):
        """
        更新当前面板对象
        Keyword arguments:
        now_map -- 当前地图
        start_index -- 起始按钮id
        """
        self.now_map = now_map
        self.handle_panel.text_list = self.get_path_list()
        self.handle_panel.button_start_id = start_index
        self.handle_panel.update()
        self.end_index = self.handle_panel.end_index

    def draw(self):
        """绘制面板"""
        self.handle_panel.draw()
        self.return_list = self.handle_panel.return_list


class ScenePathNameMoveDraw:
    """
    按场景路径绘制场景名和移动按钮
    Keyword arguments:
    text -- 场景路径
    width -- 最大宽度
    is_button -- 绘制按钮
    num_button -- 绘制数字按钮
    button_id -- 数字按钮的id
    """

    def __init__(
        self, text: List[str], width: int, is_button: bool, num_button: bool, button_id: int
    ):
        """初始化绘制对象"""
        self.draw_text = ""
        """ 场景路径绘制的文本 """
        self.width: int = width
        """ 最大宽度 """
        self.is_button: bool = is_button
        """ 绘制按钮 """
        self.num_button: bool = num_button
        """ 绘制数字按钮 """
        self.button_id: int = button_id
        """ 数字按钮的id """
        self.button_return: str = str(button_id)
        """ 按钮返回值 """
        self.draw_list: List[draw.NormalDraw] = []
        """ 绘制的对象列表 """
        self.scene_path: List[str] = text
        """ 对应场景路径 """
        path_text = attr_text.get_scene_path_text(text)
        character_data: game_type.Character = cache.character_data[0]
        character_scene_str = map_handle.get_map_system_path_str_for_list(character_data.position)
        path_str = map_handle.get_map_system_path_str_for_list(text)
        move_time = 0
        if path_str in map_handle.scene_move_time[character_scene_str]:
            move_time = map_handle.scene_move_time[character_scene_str][path_str]
        button_text = _("[{path_text}:{move_time}分钟]").format(
            path_text=path_text, move_time=move_time
        )
        button_text = text_handle.align(button_text, text_width=width)
        name_draw = draw.Button(button_text, path_text, cmd_func=self.move_now)
        self.draw_text = button_text
        name_draw.width = width
        self.now_draw = name_draw
        """ 绘制的对象 """
        self.button_return = path_text

    def draw(self):
        """绘制对象"""
        self.now_draw.draw()

    def move_now(self):
        """
        控制角色移动至指定场景
        Keyword arguments:
        scene_path -- 目标场景路径
        """
        py_cmd.clr_cmd()
        line_feed.draw()
        cache.wframe_mouse.w_frame_skip_wait_mouse = 1
        character_move.own_charcter_move(self.scene_path)


class SocialSceneNamePanel:
    """
    绘制社交对象所在场景快捷寻路按钮列表
    Keyword arguments:
    now_map -- 地图路径
    width -- 绘制宽度
    """

    def __init__(self, now_map: List[str], width: int):
        self.width: int = width
        """ 绘制的最大宽度 """
        self.now_map: List[str] = now_map
        """ 当前查看的地图坐标 """
        self.return_list: List[str] = []
        """ 当前面板的按钮返回 """
        character_data: game_type.Character = cache.character_data[0]
        social_data = value_handle.sorted_dict_for_values(character_data.social_contact_data)
        self.handle_panel = panel.PageHandlePanel(
            [k for k in social_data if social_data[k]],
            SocialSceneNameDraw,
            20,
            2,
            self.width,
            1,
        )
        self.end_index = self.handle_panel.end_index
        """ 结束按钮id """

    def update(self, now_map: List[str], start_index: int):
        """
        更新当前面板对象
        Keyword arguments:
        now_map -- 当前地
        start_index -- 起始按钮id
        """
        self.now_map = now_map
        self.handle_panel.button_start_id = start_index
        self.handle_panel.update()
        self.end_index = self.handle_panel.end_index

    def draw(self):
        """绘制面板"""
        self.handle_panel.draw()
        self.return_list = self.handle_panel.return_list


class SocialSceneNameDraw:
    """
    按角色id绘制角色所在场景名和移动按钮
    Keyword arguments:
    text -- 角色id
    width -- 最大宽度
    is_button -- 绘制按钮
    num_button -- 绘制数字按钮
    button_id -- 数字按钮的id
    """

    def __init__(self, text: int, width: int, is_button: bool, num_button: bool, button_id: int):
        """初始化绘制对象"""
        self.draw_text = ""
        """ 场景路径绘制文本 """
        self.width: int = width
        """ 最大宽度 """
        self.is_button: bool = is_button
        """ 绘制按钮 """
        self.num_button: bool = num_button
        """ 绘制数字按钮 """
        self.button_id: int = button_id
        """ 数字按钮的id """
        self.button_return: str = str(button_id)
        """ 按钮返回值 """
        self.draw_list: List[draw.NormalDraw] = []
        """ 绘制的对象列表 """
        self.target_character_id: int = text
        character_data: game_type.Character = cache.character_data[text]
        self.scene_path: List[str] = character_data.position
        """ 角色所在场景 """
        path_text = attr_text.get_scene_path_text(self.scene_path)
        player_data: game_type.Character = cache.character_data[0]
        player_scene_str = map_handle.get_map_system_path_str_for_list(player_data.position)
        path_str = map_handle.get_map_system_path_str_for_list(self.scene_path)
        social_type = 5
        if text in player_data.social_contact_data:
            social_type = player_data.social_contact_data[text]
        type_text = game_config.config_social_type[social_type].name
        move_time = 0
        if path_str in map_handle.scene_move_time[player_scene_str]:
            move_time = map_handle.scene_move_time[player_scene_str][path_str]
        button_text = _("[{character_name}:{social_contact}|{path_text}:{move_time}分钟]").format(
            character_name=character_data.name,
            social_contact=type_text,
            path_text=path_text,
            move_time=move_time,
        )
        button_text = text_handle.align(button_text, text_width=width)
        name_draw = draw.Button(button_text, character_data.name, cmd_func=self.move_now)
        name_draw.width = width
        self.now_draw = name_draw
        """ 绘制的对象 """
        self.button_return = character_data.name

    def draw(self):
        """绘制对象"""
        self.now_draw.draw()

    def move_now(self):
        """
        控制角色移动至指定场景
        Keyword arguments:
        scene_path -- 目标场景路径
        """
        py_cmd.clr_cmd()
        line_feed.draw()
        cache.wframe_mouse.w_frame_skip_wait_mouse = 1
        character_move.own_move_to_character_scene(self.target_character_id)


class CollectionSceneNamePanel:
    """
    绘制收藏对象所在场景快捷寻路按钮列表
    Keyword arguments:
    now_map -- 地图路径
    width -- 绘制宽度
    """

    def __init__(self, now_map: List[str], width: int):
        self.width: int = width
        """ 绘制的最大宽度 """
        self.now_map: List[str] = now_map
        """ 当前查看的地图坐标 """
        self.return_list: List[str] = []
        """ 当前面板的按钮返回 """
        character_data: game_type.Character = cache.character_data[0]
        self.handle_panel = panel.PageHandlePanel(
            list(character_data.collection_character),
            SocialSceneNameDraw,
            10,
            2,
            self.width,
            1,
        )
        self.end_index = self.handle_panel.end_index
        """ 结束按钮id """

    def update(self, now_map: List[str], start_index: int):
        """
        更新当前面板对象
        Keyword arguments:
        now_map -- 当前地图
        start_index -- 起始按钮id
        """
        self.now_map = now_map
        self.handle_panel.button_start_id = start_index
        self.handle_panel.update()
        self.end_index = self.handle_panel.end_index

    def draw(self):
        """绘制面板"""
        self.handle_panel.draw()
        self.return_list = self.handle_panel.return_list
