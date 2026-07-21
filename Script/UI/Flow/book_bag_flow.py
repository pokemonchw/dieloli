from Script.Design import handle_panel, constant
from Script.UI.Panel import book_bag_panel
from Script.Config import normal_config

width = normal_config.config_normal.text_width
""" 屏幕宽度 """


@handle_panel.add_panel(constant.Panel.BOOK_BAG)
def book_bag_flow():
    """书籍背包面板"""
    now_panel = book_bag_panel.BookBagPanel(width)
    now_panel.draw()
