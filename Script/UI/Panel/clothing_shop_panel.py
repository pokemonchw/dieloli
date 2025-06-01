from typing import Tuple
from types import FunctionType
from uuid import UUID
from Script.Core import (
    cache_control,
    game_type,
    get_text,
    text_handle,
    flow_handle,
    py_cmd,
)
from Script.Design import constant
from Script.UI.Model import panel, draw
from Script.UI.Panel.see_clothing_info_panel import ClothingDescribeDraw
from Script.Config import game_config, normal_config

cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
_: FunctionType = get_text._
""" 翻译api """
line_feed = draw.NormalDraw()
""" 换行绘制对象 """
line_feed.text = "\n"
line_feed.width = 1
window_width: int = normal_config.config_normal.text_width
""" 窗体宽度 """


class ClothingShopPanel:
    """
    用于查看服装商店界面面板对象
    Keyword arguments:
    width -- 绘制宽度
    """

    def __init__(self, width: int):
        """初始化绘制对象"""
        self.width: int = width
        """ 绘制的最大宽度 """
        self.clothing_type = 0
        """ 当前绘制的服装类型 """
        character_data: game_type.Character = cache.character_data[0]
        self.clothing_sex_type = 0
        """ 当前绘制的富 """
        if character_data.sex in {1, 3}:
            self.clothing_sex_type = 1
        self.handle_panel: panel.PageHandlePanel = None
        """ 当前服装列表控制面板 """

    def draw(self):
        """绘制对象"""
        title_draw = draw.TitleLineDraw(_("服装超市"), self.width)
        sex_type_list = [_("男"), _("女"), _("通用")]
        self.handle_panel = panel.PageHandlePanel(
            [], SeeClothingListByClothingNameDraw, 10, 5, self.width, 1, 1, 0
        )
        while 1:
            return_list = []
            py_cmd.clr_cmd()
            title_draw.draw()
            now_cid_list = []
            if (
                self.clothing_type in game_config.config_clothing_type_sex_type_data
                and self.clothing_sex_type
                in game_config.config_clothing_type_sex_type_data[self.clothing_type]
            ):
                now_type_cid_set = game_config.config_clothing_type_sex_type_data[
                    self.clothing_type
                ][self.clothing_sex_type]
                for cid in now_type_cid_set:
                    if cid in cache.clothing_shop and len(cache.clothing_shop[cid]):
                        now_cid_list.append(cid)
            for clothing_type in game_config.config_clothing_type:
                clothing_type_config = game_config.config_clothing_type[clothing_type]
                if clothing_type == self.clothing_type:
                    now_draw = draw.CenterDraw()
                    now_draw.text = clothing_type_config.name
                    now_draw.style = "onbutton"
                    now_draw.width = self.width / len(game_config.config_clothing_type)
                    now_draw.draw()
                else:
                    now_draw = draw.CenterButton(
                        f"[{clothing_type_config.name}]",
                        clothing_type_config.name,
                        self.width / len(game_config.config_clothing_type),
                        cmd_func=self.change_clothing_type,
                        args=(clothing_type,),
                    )
                    now_draw.draw()
                    return_list.append(now_draw.return_text)
            line_feed.draw()
            line = draw.LineDraw("+", self.width)
            line.draw()
            for sex_type in range(3):
                sex_type_text = sex_type_list[sex_type]
                if sex_type == self.clothing_sex_type:
                    now_draw = draw.CenterDraw()
                    now_draw.text = sex_type_text
                    now_draw.style = "onbutton"
                    now_draw.width = self.width / 3
                    now_draw.draw()
                else:
                    now_draw = draw.CenterButton(
                        f"[{sex_type_text}]",
                        sex_type_text,
                        self.width / 3,
                        cmd_func=self.change_sex_type,
                        args=(sex_type,),
                    )
                    now_draw.draw()
                    return_list.append(now_draw.return_text)
            line_feed.draw()
            line = draw.LineDraw("+", self.width)
            line.draw()
            if len(now_cid_list):
                self.handle_panel.text_list = now_cid_list
                self.handle_panel.update()
                self.handle_panel.draw()
                return_list.extend(self.handle_panel.return_list)
            else:
                now_text = _("空无一物")
                now_draw = draw.CenterDraw()
                now_draw.text = now_text
                now_draw.width = self.width
                now_draw.draw()
                line_feed.draw()
            line = draw.LineDraw("-", self.width)
            line.draw()
            back_draw = draw.CenterButton(_("[返回]"), _("返回"), window_width)
            back_draw.draw()
            line_feed.draw()
            return_list.append(back_draw.return_text)
            yrn = flow_handle.askfor_all(return_list)
            if yrn == back_draw.return_text:
                cache.now_panel_id = constant.Panel.IN_SCENE
                break

    def change_clothing_type(self, clothing_type: int):
        """
        切换当前面板显示的服装类型
        Keyword arguments:
        clothing_type -- 要切换的服装类型
        """
        self.clothing_type = clothing_type

    def change_sex_type(self, sex_type: int):
        """
        切换当前面板的性别类型
        Keyword arguments:
        sex_type -- 要切换的性别类型
        """
        self.clothing_sex_type = sex_type


class SeeClothingListByClothingNameDraw:
    """
    点击后可以查看服装列表的服装名字按钮对象
    Keyword arguments:
    text -- 服装cid
    width -- 最大宽度
    is_button -- 绘制按钮
    num_button -- 绘制数字按钮
    button_id -- 数字按钮id
    """

    def __init__(self, text: int, width: int, is_button: bool, num_button: bool, button_id: int):
        """初始化绘制对象"""
        self.text: int = text
        """ 服装cid """
        self.draw_text: str = ""
        """ 服装名字绘制文本 """
        self.width: int = width
        """ 最大宽度 """
        self.num_button: bool = num_button
        """ 绘制数字按钮 """
        self.button_id: int = button_id
        """ 数字按钮的id """
        self.button_return: str = str(button_id)
        """ 按钮的返回值 """
        self.clothing_config = game_config.config_clothing_tem[self.text]
        """ 服装配置数据 """
        self.clothing_name = self.clothing_config.name
        """ 当前服装名字 """
        name_draw = draw.NormalDraw()
        if is_button:
            if num_button:
                index_text = text_handle.id_index(button_id)
                button_text = f"{index_text}{self.clothing_name}"
                name_draw = draw.LeftButton(
                    button_text,
                    self.button_return,
                    self.width,
                    cmd_func=self.see_clothing_shop_clothing_list,
                )
            else:
                button_text = f"[{self.text}]"
                name_draw = draw.CenterButton(
                    button_text,
                    self.text,
                    self.width,
                    cmd_func=self.see_clothing_shop_clothing_list,
                )
                self.button_return = text
            self.draw_text = button_text
        else:
            name_draw = draw.CenterDraw()
            name_draw.text = f"[{self.text}]"
            name_draw.width = self.width
            self.draw_text = name_draw.text
        self.now_draw = name_draw
        """ 绘制的对象 """

    def draw(self):
        """绘制对象"""
        self.now_draw.draw()

    def see_clothing_shop_clothing_list(self):
        """按服装名字显示服装商店的服装列表"""
        py_cmd.clr_cmd()
        info_draw = ClothingDescribeDraw(self.text, window_width)
        info_draw.draw()
        title_draw = draw.TitleLineDraw(self.clothing_config.name, window_width)
        handle_panel = panel.ClothingPageHandlePanel(
            [], BuyClothingByClothingNameDraw, 10, 1, window_width, 1, 1
        )
        """ 页面控制对象 """
        while 1:
            return_list = []
            line_feed.draw()
            clothing_list = list(cache.clothing_shop[self.text].keys())
            if len(clothing_list):
                title_draw.draw()
                now_list = [(self.text, i) for i in clothing_list]
                handle_panel.text_list = now_list
                handle_panel.update()
                handle_panel.update_button()
                handle_panel.draw()
                return_list.extend(handle_panel.return_list)
                line_feed.draw()
                line = draw.LineDraw("-", window_width)
                line.draw()
                back_draw = draw.CenterButton(_("[返回]"), _("返回"), window_width)
                back_draw.draw()
                line_feed.draw()
                return_list.append(back_draw.return_text)
                yrn = flow_handle.askfor_all(return_list)
                py_cmd.clr_cmd()
                if yrn == back_draw.return_text:
                    break
            else:
                now_text = _("空无一物")
                now_draw = draw.WaitDraw()
                now_draw.text = now_text
                now_draw.width = self.width
                now_draw.draw()
                line_feed.draw()
                break


class BuyClothingByClothingNameDraw:
    def __init__(self, text: Tuple, width: int, is_button: bool, num_button: bool, button_id: int):
        """初始化绘制对象"""
        self.text: UUID = text[1]
        """ 服装id """
        self.clothing_type: int = text[0]
        """ 服装类型 """
        self.width: int = width
        """ 绘制宽度 """
        self.draw_text: str = ""
        """ 服装缩略信息绘制文本 """
        self.is_button: bool = is_button
        """ 绘制按钮 """
        self.num_button: bool = num_button
        """ 绘制数字按钮 """
        self.button_id: int = button_id
        """ 数字按钮的id """
        self.button_return: str = str(button_id)
        """ 按钮返回值 """
        now_id_text = ""
        now_id_text = text_handle.id_index(self.button_id)
        fix_width = self.width - len(now_id_text)
        self.clothing_data: game_type.Clothing = cache.clothing_shop[self.clothing_type][self.text]
        """ 当前服装数据 """
        value_dict = {
            _("可爱"): self.clothing_data.sweet,
            _("性感"): self.clothing_data.sexy,
            _("帅气"): self.clothing_data.handsome,
            _("清新"): self.clothing_data.fresh,
            _("典雅"): self.clothing_data.elegant,
            _("清洁"): self.clothing_data.cleanliness,
            _("保暖"): self.clothing_data.warm,
        }
        describe_list = [_("可爱的"), _("性感的"), _("帅气的"), _("清新的"), _("典雅的"), _("清洁的"), _("保暖的")]
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
        id_text = f"{now_id_text} {clothing_name}"
        self.text_list: List[str] = [id_text, value_text]
        """ 绘制的文本列表 """
        self.id_width: int = text_handle.get_text_index(id_text)
        """ id部分的绘制宽度 """
        self.draw_text = " " * self.width

    def draw(self):
        """绘制对象"""
        self.text_list[1] = text_handle.align(
            self.text_list[1], "center", text_width=self.width - self.id_width
        )
        text_width = text_handle.get_text_index(self.text_list[0])
        self.text_list[0] += " " * (self.id_width - text_width)
        now_text = f"{self.text_list[0]}{self.text_list[1]}"
        now_text = now_text.rstrip()
        now_text = text_handle.align(now_text, "center", text_width=self.width)
        now_draw = draw.Button(now_text, str(self.button_id), cmd_func=self.buy_clothing)
        now_draw.width = self.width
        now_draw.draw()

    def buy_clothing(self):
        """购买服装"""
        py_cmd.clr_cmd()
        character_data: game_type.Character = cache.character_data[0]
        character_data.clothing.setdefault(self.clothing_type, {})
        character_data.clothing[self.clothing_type][self.text] = self.clothing_data
        del cache.clothing_shop[self.clothing_type][self.text]
