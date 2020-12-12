from typing import List
from types import FunctionType
from Script.UI.Moudle import draw
from Script.Core import get_text, cache_control, game_type, flow_handle, value_handle, text_handle, constant
from Script.Design import map_handle, attr_text, character_move

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
        """ 初始化绘制对象 """
        self.width: int = width
        """ 绘制的最大宽度 """
        character_data: game_type.Character = cache.character_data[0]
        self.now_map: List[str] = map_handle.get_map_for_path(character_data.position)
        """ 当前查看的地图坐标 """

    def draw(self):
        """ 绘制对象 """
        while 1:
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
            for now_draw_line in now_draw_list.draw_text:
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
                            draw_text.text, draw_text.text, cmd_func=self.move_now, args=(scene_path,)
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
            if len(scene_path_list):
                line = draw.LineDraw(".", self.width)
                line.draw()
                message_draw = draw.NormalDraw()
                message_draw.text = _("你可以从这里前往:\n")
                message_draw.width = self.width
                message_draw.draw()
                draw_list = []
                for scene in scene_path_list:
                    load_scene_data = map_handle.get_scene_data_for_map(map_path_str, scene)
                    now_scene_path = map_handle.get_map_system_path_for_str(load_scene_data.scene_path)
                    now_draw = draw.CenterButton(
                        f"[{load_scene_data.scene_name}]",
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
            if len(scene_id_list):
                line = draw.LineDraw(".", self.width)
                line.draw()
                message_draw = draw.NormalDraw()
                message_draw.text = _("场景名列表:\n")
                message_draw.width = self.width
                message_draw.draw()
                draw_list = []
                for scene_id in scene_id_list:
                    load_scene_data = map_handle.get_scene_data_for_map(map_path_str, scene_id)
                    now_scene_path = map_handle.get_map_system_path_for_str(load_scene_data.scene_path)
                    now_id_text = f"{scene_id}:{load_scene_data.scene_name}"
                    now_draw = draw.LeftButton(
                        now_id_text, now_id_text, self.width, cmd_func=self.move_now, args=(now_scene_path,)
                    )
                    return_list.append(now_draw.return_text)
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
            line = draw.LineDraw("=", self.width)
            line.draw()
            now_index = len(scene_id_list)
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
            if yrn == back_button.return_text:
                cache.now_panel_id = constant.Panel.IN_SCENE
                break

    def up_map(self):
        """ 将当前地图切换为上级地图 """
        up_map_path = map_handle.get_map_for_path(self.now_map)
        self.now_map = up_map_path

    def down_map(self):
        """ 将当前地图切换为下级地图 """
        character_position = cache.character_data[0].position
        down_map_scene_id = map_handle.get_map_scene_id_for_scene_path(self.now_map, character_position)
        self.now_map.append(down_map_scene_id)

    def move_now(self, scene_path: List[str]):
        """
        控制角色移动至指定场景
        Keyword arguments:
        scene_path -- 目标场景路径
        """
        character_move.own_charcter_move(scene_path)


class SceneNameMoveButton:
    """
    场景名绘制通行按钮对象
    Keyword arguments:
    text -- 场景文本路径
    width -- 最大宽度
    is_button -- 绘制按钮
    num_button -- 绘制数字按钮
    button_id -- 数字按钮的id
    """

    def __init__(self, text: tuple, width: int, is_button: bool, num_button: bool, button_id: int):
        """ 初始化绘制对象 """
        self.scene_path: str = text[0]
        """ 场景的文本路径 """
        self.map_path: str = text[1]
        """ 场景所在地图文本路径 """
        self.draw_text: str = ""
        """ 场景名绘制文本 """
        self.width: int = width
        """ 最大宽度 """
        self.is_button: bool = is_button
        """ 绘制按钮 """
        self.num_button: bool = num_button
        """ 绘制数字按钮 """
