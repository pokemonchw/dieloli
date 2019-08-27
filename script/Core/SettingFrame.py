from script.Core import TextLoading
from tkinter import Tk

def openSettingFrame():
    '''
    打开设置菜单
    '''
    root = Tk()
    titleText = TextLoading.getTextData(TextLoading.systemTextPath,'Setting')['TitleName']
    root.title(titleText)
