from tkinter import Tk
from Script.Core import text_loading, constant


def open_setting_frame():
    """
    打开设置菜单
    """
    root = Tk()
    title_text = text_loading.get_text_data(constant.FilePath.SYSTEM_TEXT_PATH, "Setting")["TitleName"]
    root.title(title_text)
