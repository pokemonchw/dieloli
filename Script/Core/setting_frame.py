from tkinter import Tk
from Script.Core import text_loading


def open_setting_frame():
    """
    打开设置菜单
    """
    root = Tk()
    title_text = text_loading.get_text_data(text_loading.system_text_path, "Setting")[
        "TitleName"
    ]
    root.title(title_text)
