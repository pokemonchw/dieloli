from types import FunctionType
from Script.UI.Moudle import draw, panel
from Script.UI.Panel import game_info_panel, see_character_info_panel
from Script.Core import get_text, cache_control, game_type, flow_handle, text_handle,value_handle
from Script.Design import attr_text, map_handle

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
_: FunctionType = get_text._
""" 翻译api """
line_feed = draw.NormalDraw()
""" 换行绘制对象 """
line_feed.text = "\n"
line_feed.width = 1


class InScenePanel:
    """
    用于查看场景互动界面面板对象
    Keyword arguments:
    width -- 绘制宽度
    """

    def __init__(self, width: int):
        """ 初始化绘制对象 """
        self.width: int = width
        """ 绘制的最大宽度 """

    def draw(self):
        """ 绘制对象 """
        title_draw = draw.TitleLineDraw(_("场景"), self.width)
        character_data: game_type.Character = cache.character_data[0]
        scene_path_str = map_handle.get_map_system_path_str_for_list(character_data.position)
        scene_data: game_type.Scene = cache.scene_data[scene_path_str]
        if not character_data.target_character_id and len(scene_data.character_list):
            character_list = list(scene_data.character_list)
            character_list.remove(0)
            character_data.target_character_id = character_list[0]
        game_time_draw = game_info_panel.GameTimeInfoPanel(self.width / 2)
        game_time_draw.now_draw.width = len(game_time_draw)
        position_text = attr_text.get_scene_path_text(character_data.position)
        now_position_text = _("当前位置:") + position_text
        now_position_draw = draw.NormalDraw()
        now_position_draw.text = now_position_text
        now_position_draw.width = self.width - len(game_time_draw)
        meet_draw = draw.NormalDraw()
        meet_draw.text = _("你在这里遇到了:")
        meet_draw.width = self.width
        character_list = list(scene_data.character_list)
        character_list.remove(0)
        character_handle_panel = panel.PageHandlePanel(
            character_list,
            see_character_info_panel.SeeCharacterInfoByNameDrawInScene,
            10,
            5,
            self.width,
            1,
            0,
            null_button_text=character_data.target_character_id,
        )
        while 1:
            character_handle_panel.null_button_text = character_data.target_character_id
            line_feed.draw()
            title_draw.draw()
            game_time_draw.draw()
            now_position_draw.draw()
            line_feed.draw()
            ask_list = []
            if len(scene_data.character_list):
                meet_draw.draw()
                line_feed.draw()
                character_handle_panel.update()
                character_handle_panel.draw()
                ask_list.extend(character_handle_panel.return_list)
            character_info_draw_list = []
            if character_data.target_character_id:
                character_head_draw = see_character_info_panel.CharacterInfoHead(character_data.cid,self.width / 2)
                target_head_draw = see_character_info_panel.CharacterInfoHead(character_data.target_character_id,self.width/2)
                character_head_draw_list = [y for x in character_head_draw.draw_list for y in x]
                character_head_draw_list[0].text += " " + character_head_draw_list[2].text
                del character_head_draw_list[2]
                for value in character_head_draw_list:
                    if not isinstance(value,draw.InfoBarDraw):
                        value.text = text_handle.align(value.text,"center",0,1,self.width / 2)
                target_head_draw_list = [y for x in target_head_draw.draw_list for y in x]
                target_head_draw_list[0].text += " " + target_head_draw_list[2].text
                del target_head_draw_list[2]
                for value in target_head_draw_list:
                    if not isinstance(value,draw.InfoBarDraw):
                        value.text = text_handle.align(value.text,"center",0,1,self.width / 2)
                character_info_draw_list = list(zip(character_head_draw_list,target_head_draw_list))
            else:
                character_head_draw = see_character_info_panel.CharacterInfoHead(character_data.cid,self.width)
                character_info_draw_list = character_head_draw.draw_list
            for value_tuple in character_info_draw_list:
                for value in value_tuple:
                    if isinstance(value,draw.InfoBarDraw):
                        value.scale = 0.8
                        value.width = self.width / 2
                        value.set(value.bar_id,value.max_value,value.value,value.text)
                    value.draw()
                line_feed.draw()
            target_id = character_data.target_character_id
            flow_handle.askfor_all(ask_list)
