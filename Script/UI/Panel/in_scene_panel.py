from typing import List
from types import FunctionType
import datetime
import random
from Script.UI.Model import draw, panel
from Script.UI.Panel import game_info_panel, see_character_info_panel
from Script.UI.Panel import see_character_info_panel
from Script.Core import (
    get_text, cache_control, game_type,
    flow_handle, text_handle, value_handle,
    py_cmd, io_init,
)
from Script.Design import (
    attr_text, map_handle, handle_instruct,
    handle_premise, constant, game_time,
    handle_achieve, course,
)
from Script.Config import game_config, normal_config, config_def

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
        """初始化绘制对象"""
        self.width: int = width
        """ 绘制的最大宽度 """

    def draw(self):
        """绘制对象"""
        title_draw = draw.TitleLineDraw(_("场景"), self.width)
        character_data: game_type.Character = cache.character_data[0]
        scene_path_str = map_handle.get_map_system_path_str_for_list(character_data.position)
        scene_data: game_type.Scene = cache.scene_data[scene_path_str]
        character_handle_panel = panel.PageHandlePanel(
            [],
            see_character_info_panel.SeeCharacterInfoByNameDrawInScene,
            10,
            5,
            self.width,
            1,
            0,
            len(constant.handle_instruct_name_data),
            null_button_text=character_data.target_character_id,
        )
        old_character_set = set()
        is_collection = cache.is_collection
        while 1:
            py_cmd.clr_cmd()
            character_data: game_type.Character = cache.character_data[0]
            scene_path_str = map_handle.get_map_system_path_str_for_list(character_data.position)
            scene_data: game_type.Scene = cache.scene_data[scene_path_str]
            if character_data.dead:
                cache.wframe_mouse.w_frame_skip_wait_mouse = 0
                now_draw = draw.LineFeedWaitDraw()
                cause_of_death_config = game_config.config_cause_of_death[character_data.cause_of_death]
                death_time = game_time.get_date_text(character_data.behavior.start_time)
                now_draw.text = _("已死亡! 死因:") + f"{cause_of_death_config.name} " + death_time
                now_draw.width = self.width
                now_draw.draw()
                cache_control.achieve.first_dead = True
                handle_achieve.check_all_achieve()
                now_draw.text = _("重新开始新的人生吧")
                now_draw.width = self.width
                now_draw.draw()
                cache.now_panel_id = constant.Panel.TITLE
                break
            character_set = scene_data.character_list.copy()
            character_set.remove(0)
            refresh_character_list_judge = False
            if len(old_character_set) == 0 or old_character_set != character_set or is_collection != cache.is_collection:
                old_character_set = character_set.copy()
                refresh_character_list_judge = True
                is_collection = cache.is_collection
            if refresh_character_list_judge:
                if cache.is_collection:
                    character_list = [
                        i for i in character_set if i in character_data.collection_character
                    ]
                else:
                    character_list = list(character_set)
                live_character_dict = {}
                dead_character_dict = {}
                for now_character in character_list:
                    now_character_data: game_type.Character = cache.character_data[now_character]
                    if now_character_data.state == constant.CharacterStatus.STATUS_DEAD:
                        if now_character in character_data.favorability:
                            dead_character_dict[now_character] = character_data.favorability[
                                now_character
                            ]
                        else:
                            dead_character_dict[now_character] = 0
                    else:
                        if now_character in character_data.favorability:
                            live_character_dict[now_character] = character_data.favorability[
                                now_character
                            ]
                        else:
                            live_character_dict[now_character] = 0
                live_character_dict = value_handle.sorted_dict_for_values(live_character_dict)
                live_character_list = list(live_character_dict.keys())
                live_character_list.reverse()
                dead_character_dict = value_handle.sorted_dict_for_values(dead_character_dict)
                dead_character_list = list(dead_character_dict.keys())
                dead_character_list.reverse()
                character_list = live_character_list + dead_character_list
                if character_data.target_character_id not in scene_data.character_list:
                    character_data.target_character_id = -1
                if character_data.target_character_id == -1 and character_list:
                    character_data.target_character_id = character_list[0]
                elif character_data.target_character_id not in {-1, 0} and character_list:
                    if character_data.target_character_id in character_list:
                        character_list.remove(character_data.target_character_id)
                        new_character_list = [character_data.target_character_id]
                        new_character_list.extend(character_list)
                        character_list = new_character_list
                character_handle_panel.text_list = character_list
            game_time_draw = game_info_panel.GameTimeInfoPanel(self.width / 2)
            game_time_draw.now_draw.width = len(game_time_draw)
            position_text = attr_text.get_scene_path_text(character_data.position)
            now_position_text = _("当前位置:") + position_text
            now_position_draw = draw.NormalDraw()
            now_position_draw.text = now_position_text
            now_position_draw.width = self.width - len(game_time_draw)
            money_draw = draw.NormalDraw()
            money_draw.text = _(" 金钱:") + str(round(character_data.money, 2))
            money_draw.width = self.width
            meet_draw = draw.NormalDraw()
            meet_draw.text = _("你在这里遇到了:")
            meet_draw.width = self.width
            see_instruct_panel = SeeInstructPanel(100)
            cache.wframe_mouse.w_frame_skip_wait_mouse = 0
            if cache.now_panel_id != constant.Panel.IN_SCENE:
                break
            character_handle_panel.null_button_text = character_data.target_character_id
            line_feed.draw()
            title_draw.draw()
            game_time_draw.draw()
            now_position_draw.draw()
            money_draw.draw()
            line_feed.draw()
            ask_list = []
            if character_list:
                meet_draw.draw()
                line_feed.draw()
                character_handle_panel.update()
                character_handle_panel.draw()
                ask_list.extend(character_handle_panel.return_list)
                line_draw = draw.LineDraw("-.-", self.width)
                line_draw.draw()
            character_info_draw_list = []
            if character_data.target_character_id != -1:
                target_data = cache.character_data[character_data.target_character_id]
                character_head_draw = see_character_info_panel.CharacterInfoHead(
                    character_data.cid, self.width
                )
                target_head_draw = see_character_info_panel.CharacterInfoHead(
                    character_data.target_character_id, self.width
                )
                character_head_draw_list = [y for x in character_head_draw.draw_list for y in x]
                character_head_draw_list[0].text += " " + character_head_draw_list[2].text
                del character_head_draw_list[2]
                target_head_draw_list = [y for x in target_head_draw.draw_list for y in x]

                target_head_draw_list[0].text += " " + target_head_draw_list[2].text
                del target_head_draw_list[2]
                character_info_draw_list = list(
                    zip(character_head_draw_list, target_head_draw_list)
                )
            else:
                character_head_draw = see_character_info_panel.CharacterInfoHead(
                    character_data.cid, self.width
                )
                character_info_draw_list = character_head_draw.draw_list
            for value_tuple in character_info_draw_list:
                for value in value_tuple:
                    value.draw()
                line_feed.draw()
            all_character_figure_draw_list = []
            line = draw.LineDraw("_", self.width)
            line.draw()
            son_line = draw.LineDraw(".", self.width)
            stature_info_title_draw = draw.NormalDraw()
            stature_info_title_draw.width = self.width
            stature_info_title_draw.text = _("口 身材信息: ")
            stature_info_title_draw.draw()
            if cache.in_scene_panel_switch.stature_switch:
                stature_switch_button = draw.Button(_("[-关-]"), _("关闭身材信息面板"),cmd_func=self.change_stature_panel_swicth)
                stature_switch_button.width = self.width
                stature_switch_button.draw()
                ask_list.append(stature_switch_button.return_text)
                line_feed.draw()
                son_line.draw()
                if character_data.target_character_id != -1:
                    character_stature_info = see_character_info_panel.CharacterStatureInfoText(0, self.width)
                    character_measurements_info = see_character_info_panel.CharacterMeasurementsText(0, self.width)
                    character_figure_info_list = character_stature_info.info_list + character_measurements_info.info_list
                    character_figure_info_list.insert(3, "")
                    character_figure_draw = panel.LeftDrawTextListPanel()
                    character_figure_draw.set(character_figure_info_list, self.width / 2 - 1, 4)
                    target_stature_info = see_character_info_panel.CharacterStatureInfoText(
                        character_data.target_character_id, self.width)
                    target_measurements_info = see_character_info_panel.CharacterMeasurementsText(
                        character_data.target_character_id, self.width)
                    target_figure_info_list = target_stature_info.info_list + target_measurements_info.info_list
                    target_figure_info_list.insert(3, "")
                    target_figure_draw = panel.LeftDrawTextListPanel()
                    target_figure_draw.set(target_figure_info_list, self.width / 2 - 1, 4)
                    now_line = max(len(character_figure_draw.draw_list), len(target_figure_draw.draw_list))
                    for i in range(now_line):
                        c_draw = None
                        if i in range(len(character_figure_draw.draw_list)):
                            c_draw = character_figure_draw.draw_list[i]
                        else:
                            c_draw = draw.NormalDraw()
                            c_draw.text = " " * int(self.width / 2)
                            c_draw.width = self.width / 2
                        t_draw = None
                        if i in range(len(target_figure_draw.draw_list)):
                            t_draw = target_figure_draw.draw_list[i]
                        else:
                            t_draw = draw.NormalDraw()
                            t_draw.text = " " * int(self.width / 2)
                            t_draw.width = self.width / 2
                        all_character_figure_draw_list.append((c_draw, t_draw))
                else:
                    character_stature_info = see_character_info_panel.CharacterStatureInfoText(0, self.width)
                    character_measurements_info = see_character_info_panel.CharacterMeasurementsText(0, self.width)
                    character_figure_info_list = character_stature_info.info_list + character_measurements_info.info_list
                    character_figure_info_list.insert(3, "")
                    character_figure_draw = panel.LeftDrawTextListPanel()
                    character_figure_draw.set(character_figure_info_list, self.width, 4)
                    all_character_figure_draw_list = character_figure_draw.draw_list
                for label in all_character_figure_draw_list:
                    if isinstance(label, tuple):
                        index = 0
                        for value in label:
                            if isinstance(value, list):
                                for value_draw in value:
                                    value_draw.draw()
                            elif not index:
                                value.draw()
                            if not index:
                                fix_draw = draw.NormalDraw()
                                fix_draw.width = 1
                                fix_draw.text = "|"
                                fix_draw.draw()
                                index = 1
                        line_feed.draw()
                    else:
                        for value in label:
                            value.draw()
                        line_feed.draw()
            else:
                stature_switch_button = draw.Button(_("[+开+]"), _("开启身材信息面板"),cmd_func=self.change_stature_panel_swicth)
                stature_switch_button.width = self.width
                stature_switch_button.draw()
                ask_list.append(stature_switch_button.return_text)
                line_feed.draw()
            line.draw()
            clothing_info_title_draw = draw.NormalDraw()
            clothing_info_title_draw.width = self.width
            clothing_info_title_draw.text = _("口 衣着信息: ")
            clothing_info_title_draw.draw()
            if cache.in_scene_panel_switch.clothing_switch:
                clothing_switch_button = draw.Button(_("[-关-]"), _("关闭衣着信息面板"),cmd_func=self.change_clothing_panel_switch)
                clothing_switch_button.width = self.width
                clothing_switch_button.draw()
                ask_list.append(clothing_switch_button.return_text)
                line_feed.draw()
                son_line.draw()
                character_clothing_draw_list = []
                if character_data.target_character_id != -1:
                    character_clothing_draw = see_character_info_panel.CharacterWearClothingList(
                        0, self.width / 2, 2
                    )
                    target_clothing_draw = see_character_info_panel.CharacterWearClothingList(
                        character_data.target_character_id, self.width / 2 - 1, 2
                    )
                    now_line = len(character_clothing_draw.draw_list)
                    if len(target_clothing_draw.draw_list) > now_line:
                        now_line = len(target_clothing_draw.draw_list)
                    for i in range(now_line):
                        c_draw = None
                        if i in range(len(character_clothing_draw.draw_list)):
                            c_draw = character_clothing_draw.draw_list[i]
                        else:
                            c_draw = draw.NormalDraw()
                            c_draw.text = " " * int(self.width / 2)
                            c_draw.width = self.width / 2
                        t_draw = None
                        if i in range(len(target_clothing_draw.draw_list)):
                            t_draw = target_clothing_draw.draw_list[i]
                        else:
                            t_draw = draw.NormalDraw()
                            t_draw.text = " " * int(self.width / 2 - 1)
                            t_draw.width = self.width / 2 - 1
                        character_clothing_draw_list.append((c_draw, t_draw))
                else:
                    character_clothing_draw_list = see_character_info_panel.CharacterWearClothingList(
                        0, self.width, 4
                    ).draw_list
                character_clothing_draw_list = character_clothing_draw_list[1:]
                for label in character_clothing_draw_list:
                    if isinstance(label, tuple):
                        index = 0
                        for value in label:
                            if isinstance(value, list):
                                for value_draw in value:
                                    value_draw.draw()
                            elif not index:
                                if isinstance(value, draw.LittleTitleLineDraw):
                                    continue
                                value.draw()
                            if not index:
                                fix_draw = draw.NormalDraw()
                                fix_draw.width = 1
                                fix_draw.text = "|"
                                fix_draw.draw()
                                index = 1
                        line_feed.draw()
                    elif isinstance(label, list):
                        for value in label:
                            value.draw()
                        line_feed.draw()
                    else:
                        label.draw()
            else:
                clothing_switch_button = draw.Button(_("[+开+]"), _("开启衣着信息面板"),cmd_func=self.change_clothing_panel_switch)
                clothing_switch_button.width = self.width
                clothing_switch_button.draw()
                line_feed.draw()
                ask_list.append(clothing_switch_button.return_text)
            character_status_draw_list = []
            line.draw()
            status_info_title_draw = draw.NormalDraw()
            status_info_title_draw.width = self.width
            status_info_title_draw.text = _("口 状态信息: ")
            status_info_title_draw.draw()
            if cache.in_scene_panel_switch.status_switch:
                status_switch_button = draw.Button(_("[-关-]"), _("关闭状态信息面板"),cmd_func=self.change_status_panel_switch)
                status_switch_button.width = self.width
                status_switch_button.draw()
                ask_list.append(status_switch_button.return_text)
                line_feed.draw()
                if character_data.target_character_id != -1:
                    character_status_draw = see_character_info_panel.SeeCharacterStatusPanel(
                        character_data.cid, self.width / 2, 3, 0
                    )
                    target_status_draw = see_character_info_panel.SeeCharacterStatusPanel(
                        character_data.target_character_id, self.width / 2 - 1, 3, 0
                    )
                    now_draw_line = draw.LineDraw(".", self.width)
                    character_status_draw_list.append(now_draw_line)
                    character_status_draw_list.extend(character_status_draw.draw_list[1:3])
                    fix_draw = draw.NormalDraw()
                    fix_draw.width = 1
                    fix_draw.text = "|"
                    character_status_draw_list.append(fix_draw)
                    character_status_draw_list.extend(target_status_draw.draw_list[1:3])
                    character_status_draw_list.append(line_feed)
                    for type_index in range(4, len(character_status_draw.draw_list)):
                        now_characer_status_draw = character_status_draw.draw_list[type_index]
                        now_target_status_draw = target_status_draw.draw_list[type_index]
                        now_type_draw = now_characer_status_draw.title_draw
                        now_type_draw.width = self.width
                        character_status_draw_list.append(now_type_draw)
                        now_line = max(
                            len(now_characer_status_draw.draw_list),
                            len(now_target_status_draw.draw_list),
                        )
                        for i in range(now_line):
                            c_draw = None
                            if i in range(len(now_characer_status_draw.draw_list)):
                                c_draw = now_characer_status_draw.draw_list[i]
                            else:
                                c_draw = draw.NormalDraw()
                                c_draw.text = " " * int(self.width / 2)
                                c_draw.width = self.width / 2
                            t_draw = None
                            if i in range(len(now_target_status_draw.draw_list)):
                                t_draw = now_target_status_draw.draw_list[i]
                            else:
                                t_draw = draw.NormalDraw()
                                t_draw.text = " " * int(self.width / 2 - 1)
                                t_draw.width = self.width / 2 - 1
                            character_status_draw_list.append((c_draw, t_draw))
                    for label in character_status_draw_list:
                        if isinstance(label, tuple):
                            index = 0
                            for value in label:
                                if isinstance(value, list):
                                    for value_draw in value:
                                        value_draw.draw()
                                elif not index:
                                    value.draw()
                                if not index:
                                    fix_draw = draw.NormalDraw()
                                    fix_draw.width = 1
                                    fix_draw.text = "|"
                                    fix_draw.draw()
                                    index = 1
                            line_feed.draw()
                        else:
                            label.draw()
                else:
                    character_status_draw = see_character_info_panel.SeeCharacterStatusPanel(
                        character_data.cid, self.width, 6, 0
                    )
                    character_status_draw.draw(has_title=False)
            else:
                status_switch_button = draw.Button(_("[+开+]"), _("开启状态信息面板"),cmd_func=self.change_status_panel_switch)
                status_switch_button.width = self.width
                status_switch_button.draw()
                line_feed.draw()
                ask_list.append(status_switch_button.return_text)
            see_instruct_panel.draw()
            ask_list.extend(see_instruct_panel.return_list)
            flow_handle.askfor_all(ask_list)
            py_cmd.clr_cmd(refresh_panel=False)

    def change_stature_panel_swicth(self):
        """ 更改身材信息面板开关状态 """
        cache.in_scene_panel_switch.stature_switch = not cache.in_scene_panel_switch.stature_switch

    def change_clothing_panel_switch(self):
        """ 更改穿着信息面板开关状态 """
        cache.in_scene_panel_switch.clothing_switch = not cache.in_scene_panel_switch.clothing_switch

    def change_status_panel_switch(self):
        """ 更改状态信息面板开关状态 """
        cache.in_scene_panel_switch.status_switch = not cache.in_scene_panel_switch.status_switch


class SeeInstructPanel:
    """
    查看操作菜单面板
    Keyword arguments:
    width -- 绘制宽度
    """

    def __init__(self, width: int):
        """初始化绘制对象"""
        self.width: int = width
        """ 最大绘制宽度 """
        self.return_list: List[str] = []
        """ 监听的按钮列表 """
        if cache.instruct_filter == {}:
            for instruct_type in game_config.config_instruct_type:
                cache.instruct_filter[instruct_type] = 0
            cache.instruct_filter[0] = 1

    def draw(self):
        """绘制操作菜单面板"""
        self.return_list = []
        now_line_feed = draw.NormalDraw()
        """ 换行绘制对象 """
        now_line_feed.text = "\n"
        now_line_feed.width = 1
        now_line_feed.draw_instruct = True
        line = draw.LineDraw("-.-", self.width)
        line.draw_instruct = True
        line.draw()
        now_premise_data = {}
        instruct_len_max = 0
        instruct_type_data = {}
        for now_type in constant.instruct_type_data:
            instruct_type_data.setdefault(now_type, [])
            for instruct in constant.instruct_type_data[now_type]:
                premise_judge = 0
                if instruct in constant.instruct_premise_data:
                    for premise in constant.instruct_premise_data[instruct]:
                        if premise in now_premise_data:
                            if now_premise_data[premise]:
                                continue
                            premise_judge = 1
                            break
                        now_premise_value = handle_premise.handle_premise(premise, 0)
                        now_premise_data[premise] = now_premise_value
                        if not now_premise_value:
                            premise_judge = 1
                            break
                instruct_name = constant.handle_instruct_name_data[instruct]
                instruct_len = text_handle.get_text_index(instruct_name)
                if (instruct_len + 5) % 2 != 0:
                    instruct_len += 1
                if instruct_len > instruct_len_max:
                    instruct_len_max = instruct_len
                if premise_judge:
                    continue
                instruct_type_data[now_type].append(instruct)
            instruct_type_data[now_type].sort()
        instruct_len_max += 5
        col = int(self.width / instruct_len_max)
        for now_type in cache.instruct_filter:
            if not normal_config.config_normal.nsfw:
                if now_type in {constant.InstructType.SEX, constant.InstructType.OBSCENITY}:
                    continue
            instruct_type_config = game_config.config_instruct_type[now_type]
            instruct_type_draw = draw.NormalDraw()
            instruct_type_draw.draw_instruct = True
            instruct_type_draw.text = instruct_type_config.name + ":"
            instruct_type_draw.width = text_handle.get_text_index(instruct_type_draw.text)
            instruct_type_draw.draw()
            if cache.instruct_filter[now_type] and now_type in constant.instruct_type_data:
                instruct_type_switch_button = draw.Button(_("[-关-]"), f"Close{now_type}Instruct",cmd_func=self.change_filter,args=(now_type,))
                instruct_type_switch_button.draw_instruct = True
                instruct_type_switch_button.width = text_handle.get_text_index(instruct_type_switch_button.text)
                instruct_type_switch_button.draw()
                self.return_list.append(instruct_type_switch_button.return_text)
                fix_draw = draw.LineDraw(".", self.width-instruct_type_draw.width-instruct_type_switch_button.width)
                fix_draw.draw_instruct = True
                fix_draw.draw()
                rows = 1
                now_instruct_list = instruct_type_data[now_type]
                cols = int(normal_config.config_normal.text_width / instruct_len_max)
                for i in range(len(now_instruct_list)):
                    instruct_id = now_instruct_list[i]
                    instruct_name = constant.handle_instruct_name_data[instruct_id]
                    instruct_name = self.change_instruct_text(instruct_name)
                    id_text = text_handle.id_index(instruct_id)
                    now_text = f"{id_text}{instruct_name}"
                    now_draw = draw.LeftButton(
                        now_text,
                        str(instruct_id),
                        int(instruct_len_max),
                        cmd_func=self.handle_instruct,
                        args=(instruct_id,),
                        draw_instruct=True,
                    )
                    now_draw.draw()
                    self.return_list.append(now_draw.return_text)
                    if i + 1 >= col and not (i + 1) % col and i + 1 != len(now_instruct_list):
                        now_line_feed.draw()
                if now_instruct_list:
                    now_line_feed.draw()
            else:
                instruct_type_switch_button = draw.Button(_("[-开-]"), f"Close{now_type}Instruct",cmd_func=self.change_filter,args=(now_type,))
                instruct_type_switch_button.width = text_handle.get_text_index(instruct_type_switch_button.text)
                instruct_type_switch_button.draw_instruct = True
                instruct_type_switch_button.draw()
                self.return_list.append(instruct_type_switch_button.return_text)
                fix_draw = draw.LineDraw(".", self.width-instruct_type_draw.width-instruct_type_switch_button.width)
                fix_draw.draw_instruct = True
                fix_draw.draw()

    @staticmethod
    def change_filter(now_type: int):
        """
        更改指令类型过滤状态
        Keyword arguments:
        now_type -- 指令类型
        """
        if cache.instruct_filter[now_type]:
            cache.instruct_filter[now_type] = 0
        else:
            cache.instruct_filter[now_type] = 1

    @staticmethod
    def handle_instruct(instruct_id: int):
        """
        处理玩家操作指令
        Keyword arguments:
        instruct_id -- 指令id
        """
        py_cmd.clr_cmd(refresh_panel=False)
        handle_instruct.handle_instruct(instruct_id)

    @staticmethod
    def change_instruct_text(instruct_text: str) -> str:
        """
        处理更新指令文本
        Keyword arguments:
        instruct_text -- 原始指令文本
        Return arguments:
        str -- 更新后的文本
        """
        if "{ClassName}" in instruct_text:
            character_data: game_type.Character = cache.character_data[0]
            end_time = 0
            school_id, phase = course.get_character_school_phase(0)
            now_time = datetime.datetime.fromtimestamp(cache.game_time, game_time.time_zone)
            now_time_value = now_time.hour * 100 + now_time.minute
            now_course_index = 0
            for session_id in game_config.config_school_session_data[school_id]:
                session_config = game_config.config_school_session[session_id]
                if session_config.start_time <= now_time_value <= session_config.end_time:
                    now_value = int(now_time_value / 100) * 60 + now_time_value % 100
                    end_value = int(session_config.end_time / 100) * 60 + session_config.end_time % 100
                    end_time = end_value - now_value + 1
                    now_course_index = session_config.session
                    break
            now_week = now_time.weekday()
            if not now_course_index:
                instruct_text = instruct_text.format(ClassName=_("自习课"))
            else:
                now_course = cache.course_time_table_data[school_id][phase][now_week][now_course_index]
                course_config:config_def.Course = game_config.config_course[now_course]
                instruct_text = instruct_text.format(ClassName=course_config.name)
        return instruct_text
