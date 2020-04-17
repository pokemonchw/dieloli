from tkinter import Label, Tk, Event
import webbrowser
from Script.Core import text_loading, game_config, constant


def go_home(event: Event):
    """
    通过浏览器打开到游戏主页的链接
    Keyword arguments:
    event -- 点击事件
    """
    webbrowser.open(r"https://github.com/pokemonchw/dieloli")


def go_license(event: Event):
    """
    通过浏览器打开协议链接
    keyword arguments:
    event -- 点击事件
    """
    webbrowser.open(r"http://creativecommons.org/licenses/by-nc-sa/2.0/")


def open_about_frame():
    """
    打开设置菜单
    """
    root = Tk()
    title_text = text_loading.get_text_data(
        constant.FilePath.SYSTEM_TEXT_PATH, "About"
    )["TitleName"]
    root.title(title_text)
    game_name = game_config.game_name
    name = Label(root, text=game_name)
    name.config(font=("Courier", 20))
    name.pack()
    link_info = text_loading.get_text_data(
        constant.FilePath.SYSTEM_TEXT_PATH, "About"
    )["GoHome"]
    link = Label(root, text=link_info, fg="blue")
    link.pack()
    link.bind("<Button-1>", go_home)
    license_info = text_loading.get_text_data(
        constant.FilePath.SYSTEM_TEXT_PATH, "About"
    )["Licenses"]
    licenese = Label(root, text=license_info, fg="blue")
    licenese.pack()
    licenese.bind("<Button-1>", go_license)
    root.mainloop()
