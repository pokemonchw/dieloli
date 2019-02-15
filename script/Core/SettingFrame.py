from script.Core import TextLoading
from tkinter import Tk

# 设置菜单
def openSettingFrame():
    root = Tk()
    titleText = TextLoading.getTextData(TextLoading.systemTextPath,'Setting')['TitleName']
    root.title(titleText)