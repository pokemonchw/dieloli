import datetime
from types import FunctionType
from Script.UI.Moudle import panel, draw
from Script.Core import (
    get_text,
    game_type,
    cache_control,
    text_handle,
    py_cmd,
    flow_handle,
)
from Script.Config import game_config, normal_config
from Script.Design import character_move, attr_text, game_time, map_handle, constant, attr_calculation

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
        self.now_gender_type = _("所有性别")
        """ 当前面板性别类型 """
        self.now_age_type = _("所有年龄")
        """ 当前面板年龄类型 """
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
            panel.PageHandlePanel(
                self.live_list, SeeCharacterStatusDraw, 10, 1, window_width, 1, 1, 0
            ),
            panel.PageHandlePanel(
                self.dead_list, SeeCharacterStatusDraw, 10, 1, window_width, 1, 1, 0
            ),
        ]
        gender_type_list = [_("所有性别"), _("男"), _("女"), _("扶她"), _("无性")]
        age_type_list = [_("所有年龄"), _("小学"), _("初中"), _("高中"), _("成年")]
        while 1:
            if cache.now_panel_id != constant.Panel.VIEW_CHARACTER_STATUS_LIST:
                break
            if self.now_panel == panel_type_list[0]:
                self.handle_panel = panel_data[0]
                now_charater_list = []
                if self.now_gender_type == _("所有性别"):
                    now_charater_list = self.live_list
                elif self.now_gender_type == _("男"):
                    now_charater_list = [character_id for character_id in self.live_list if cache.character_data[character_id].sex == 0]
                elif self.now_gender_type == _("女"):
                    now_charater_list = [character_id for character_id in self.live_list if cache.character_data[character_id].sex == 1]
                elif self.now_gender_type == _("扶她"):
                    now_charater_list = [character_id for character_id in self.live_list if cache.character_data[character_id].sex == 2]
                elif self.now_gender_type == _("无性"):
                    now_charater_list = [character_id for character_id in self.live_list if cache.character_data[character_id].sex == 3]
                if self.now_age_type == _("小学"):
                    now_charater_list = [character_id for character_id in now_charater_list if cache.character_data[character_id].age in range(7,12)]
                elif self.now_age_type == _("初中"):
                    now_charater_list = [character_id for character_id in now_charater_list if cache.character_data[character_id].age in range(13,15)]
                elif self.now_age_type == _("高中"):
                    now_charater_list = [character_id for character_id in now_charater_list if cache.character_data[character_id].age in range(16,18)]
                elif self.now_age_type == _("成年"):
                    now_charater_list = [character_id for character_id in now_charater_list if cache.character_data[character_id].age > 18]
                self.handle_panel.text_list = now_charater_list
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
            if self.now_panel == panel_type_list[0]:
                for gender_type in gender_type_list:
                    if gender_type == self.now_gender_type:
                        now_draw = draw.CenterDraw()
                        now_draw.text = f"[{gender_type}]"
                        now_draw.style = "onbutton"
                        now_draw.width = self.width / len(gender_type_list)
                        now_draw.draw()
                    else:
                        now_draw = draw.CenterButton(
                            f"[{gender_type}]",
                            gender_type,
                            self.width / len(gender_type_list),
                            cmd_func=self.change_gender_type,
                            args=(gender_type,),
                        )
                        now_draw.draw()
                        return_list.append(now_draw.return_text)
            line_feed.draw()
            line.draw()
            if self.now_panel == panel_type_list[0]:
                for age_type in age_type_list:
                    if age_type == self.now_age_type:
                        now_draw = draw.CenterDraw()
                        now_draw.text = f"[{age_type}]"
                        now_draw.style = "onbutton"
                        now_draw.width = self.width / len(age_type_list)
                        now_draw.draw()
                    else:
                        now_draw = draw.CenterButton(
                            f"[{age_type}]",
                            age_type,
                            self.width / len(age_type_list),
                            cmd_func=self.change_age_type,
                            args=(age_type,),
                        )
                        now_draw.draw()
                        return_list.append(now_draw.return_text)
            line_feed.draw()
            line.draw()
            self.handle_panel.draw()
            return_list.extend(self.handle_panel.return_list)
            back_draw = draw.CenterButton(_("[返回]"), _("返回"), window_width)
            back_draw.draw()
            line_feed.draw()
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

    def change_gender_type(self, gender_type: str):
        """
        切换当前面板显示的性别列表类型
        Keyword arguments:
        gender_type -- 要切换的性别类型
        """
        self.now_gender_type = gender_type
        py_cmd.clr_cmd()

    def change_age_type(self, age_type: str):
        """
        切换当前面板显示的年龄列表类型
        Keyword arguments:
        age_type -- 要切换的年龄类型
        """
        self.now_age_type = age_type
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
        character_id = character_data.cid
        target_text = ""
        if character_id in cache.character_target_data:
            now_target = cache.character_target_data[character_id]
            target_id = now_target.uid
            if now_target.affiliation != "":
                target_id = now_target.affiliation
            now_target_data = game_config.config_target[target_id]
            target_text = now_target_data.text
        if target_text == "":
            target_text = _("发呆中")
        position_text = attr_text.get_scene_path_text(character_data.position)
        sex_text = game_config.config_sex_tem[character_data.sex].name
        if character_data.dead:
            cause_of_death_config = game_config.config_cause_of_death[character_data.cause_of_death]
            death_time_text = _("死亡时间:")
            death_time = datetime.datetime.fromtimestamp(
                character_data.behavior.start_time, game_time.time_zone
            )
            death_time_text = f"{death_time_text}{str(death_time)}"
            button_text = f"{index_text} {character_data.name} {sex_text} {character_data.age}岁 {cause_of_death_config.name} {death_time_text} {position_text}"
        else:
            now_height = str(round(character_data.height.now_height, 2))
            now_height_text = _("身高:") + now_height + "cm"
            now_weight = str(round(character_data.weight, 2))
            now_weight_text = _("体重:") + now_weight + "kg"
            now_chest_tem_id = attr_calculation.judge_chest_group(character_data.chest.now_chest)
            now_chest_tem = game_config.config_chest[now_chest_tem_id]
            body_fat = str(round(character_data.bodyfat, 2))
            body_fat_text = _("体脂率:") + body_fat + "%"
            now_chest_text = _("罩杯:") + now_chest_tem.info
            button_text = f"{index_text} {character_data.name} {sex_text} {character_data.age}岁 {now_height_text} {now_weight_text} {body_fat_text} {now_chest_text} {target_text} {position_text}"
        player_data = cache.character_data[0]
        if self.text in player_data.collection_character:
            button_text += _(" (已收藏)")
        now_draw = draw.LeftButton(
            button_text, self.button_return, self.width, cmd_func=self.ask_for_select_character
        )
        self.now_draw = now_draw
        """ 绘制的对象 """

    def draw(self):
        """绘制对象"""
        self.now_draw.draw()

    def ask_for_select_character(self):
        """ 选择点击角色按钮的目的 """
        py_cmd.clr_cmd()
        if not self.text:
            return
        now_draw = panel.OneMessageAndSingleColumnButton()
        player_data = cache.character_data[0]
        if self.text in player_data.collection_character:
            now_draw.set([_("移动至所在场景"), _("取消收藏"), _("返回")], "准备做什么?")
        else:
            now_draw.set([_("移动至所在场景"), _("收藏"), _("返回")], "准备做什么?")
        now_draw.draw()
        return_list = now_draw.get_return_list()
        ans = flow_handle.askfor_all(return_list.keys())
        py_cmd.clr_cmd()
        now_key = return_list[ans]
        if now_key == _("移动至所在场景"):
            character_data: game_type.Character = cache.character_data[self.text]
            py_cmd.clr_cmd()
            line_feed.draw()
            cache.wframe_mouse.w_frame_skip_wait_mouse = 1
            character_move.own_move_to_character_scene(character_data.cid)
        elif now_key == _("收藏"):
            player_data.collection_character.add(self.text)
        elif now_key == _("取消收藏"):
            player_data.collection_character.remove(self.text)


