import core.TextLoading as textload
from tkinter import *

# 设置菜单
def openSettingFrame():
    root = Tk()
    titleText = textload.loadSystemText('Setting')['TitleName']
    root.title(titleText)