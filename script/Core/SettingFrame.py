import Core.TextLoading as textload
from tkinter import Tk

# 设置菜单
def openSettingFrame():
    root = Tk()
    titleText = textload.getTextData(textload.systemId,'Setting')['TitleName']
    root.title(titleText)