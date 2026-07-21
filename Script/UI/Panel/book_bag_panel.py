from uuid import UUID
from types import FunctionType
from Script.Core import get_text, game_type, cache_control, flow_handle, text_handle
from Script.UI.Model import panel, draw
from Script.Design import update, constant
from Script.Config import normal_config, game_config

_: FunctionType = get_text._
""" 翻译api """
window_width: int = normal_config.config_normal.text_width
""" 窗体宽度 """
cache: game_type.Cache = cache_control.cache
""" 游戏缓存数据 """
line_feed = draw.NormalDraw()
""" 换行绘制对象 """
line_feed.text = "\n"
line_feed.width = 1


class BookBagPanel:
    """
    用于查看书籍背包界面面板对象
    Keyword arguments:
    width -- 绘制宽度
    """

    def __init__(self, width: int):
        """初始化绘制对象"""
        self.width: int = width
        """ 绘制的最大宽度 """
        self.handle_panel: panel.PageHandlePanel = None
        """ 当前名字列表控制面板 """

    def draw(self):
        """绘制对象"""
        title_draw = draw.TitleLineDraw(_("书籍背包"), self.width)
        book_id_list = list(cache.character_data[0].book_bag)
        self.handle_panel = panel.PageHandlePanel(
            book_id_list, ReadBookDraw, 10, True, window_width, True, True, 0
        )
        while 1:
            if cache.now_panel_id != constant.Panel.BOOK_BAG:
                break
            self.handle_panel.update()
            title_draw.draw()
            return_list = []
            
            line_feed.draw()
            line = draw.LineDraw("+", self.width)
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

class ReadBookDraw:
    """
    点击后可阅读书籍的书籍名字按钮对象
    """
    def __init__(self, uid: str, width: int, is_button: bool, num_button: bool, button_id: int):
        self.uid: str = uid
        """ 书籍id """
        self.draw_text: str = ""
        """ 书籍名字绘制文本 """
        self.width: int = width
        """ 最大宽度 """
        self.num_button: bool = num_button
        """ 绘制数字按钮 """
        self.button_id: int = button_id
        """ 按钮返回值 """
        self.button_return: str = str(button_id)
        """ 按钮返回值 """
        
        book_data = game_config.config_book.get(self.uid)
        if book_data:
            book_name = book_data.name
            display_text = f"{book_name}"
        else:
            display_text = _("未知书籍")
        
        name_draw = draw.NormalDraw()
        if is_button:
            if num_button:
                index_text = text_handle.id_index(button_id)
                button_text = f"{index_text}{display_text}"
                name_draw = draw.LeftButton(
                    button_text, self.button_return, self.width, cmd_func=self.select_book
                )
            else:
                button_text = f"[{display_text}]"
                name_draw = draw.CenterButton(
                    button_text, self.uid, self.width, cmd_func=self.select_book
                )
                self.button_return = self.uid
            self.draw_text = button_text
        else:
            name_draw = draw.CenterDraw()
            name_draw.text = f"[{display_text}]"
            name_draw.width = self.width
            self.draw_text = name_draw.text
            
        self.now_draw = name_draw
        """ 绘制的对象 """

    def draw(self):
        """绘制对象"""
        self.now_draw.draw()

    def select_book(self):
        """显示书籍详情在事件面板并等待阅读确认"""
        from Script.Core import io_init, py_cmd
        
        book_data = game_config.config_book.get(self.uid)
        if book_data:
            book_name = book_data.name
            book_info = book_data.info
        else:
            book_name = _("未知书籍")
            book_info = _("没有任何信息的未知书籍。")
            
        event_text = _("你拿起了【{book_name}】。\n书籍简介：{book_info}\n").format(
            book_name=book_name, book_info=book_info
        )
        io_init.era_print(event_text, draw_type="event")
        
        while 1:
            py_cmd.clr_cmd()
            line = draw.LineDraw("-", window_width)
            line.draw()
            
            info_draw = draw.NormalDraw()
            info_draw.text = _("要阅读【{book_name}】吗？\n").format(book_name=book_name)
            info_draw.width = window_width
            info_draw.draw()
            
            return_list = []
            
            read_draw = draw.LeftButton(_("[阅读]"), _("阅读"), window_width)
            read_draw.draw()
            return_list.append(read_draw.return_text)
            line_feed.draw()
            
            cancel_draw = draw.LeftButton(_("[取消]"), _("取消"), window_width)
            cancel_draw.draw()
            return_list.append(cancel_draw.return_text)
            line_feed.draw()
            
            yrn = flow_handle.askfor_all(return_list)
            if yrn == read_draw.return_text:
                self.read_book()
                break
            elif yrn == cancel_draw.return_text:
                break

    def read_book(self):
        """阅读书籍"""
        from Script.Design import character
        character.init_character_behavior_start_time(0, cache.game_time)
        character_data: game_type.Character = cache.character_data[0]
        character_data.behavior.behavior_id = constant.Behavior.READ_BOOK
        character_data.behavior.read_book_id = self.uid
        book_data = game_config.config_book.get(self.uid)
        if book_data:
            character_data.behavior.book_name = book_data.name
        
        character_data.state = constant.CharacterStatus.STATUS_READ_BOOK
        character_data.behavior.duration = 10
        update.game_update_flow(10)
        cache.now_panel_id = constant.Panel.IN_SCENE
