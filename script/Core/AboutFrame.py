from script.Core import TextLoading, GameConfig
from tkinter import Label, Tk
import webbrowser

'''
打开主页
@event 点击事件
'''
def gohome(event):
    webbrowser.open(r"https://github.com/pokemonchw/dieloli")

'''
打开协议信息
@event 点击事件
'''
def golicense(event):
    webbrowser.open(r"http://creativecommons.org/licenses/by-nc-sa/2.0/")

'''
打开设置菜单
'''
def openAboutFrame():
    root = Tk()
    titleText = TextLoading.getTextData(TextLoading.systemTextPath, 'About')['TitleName']
    root.title(titleText)
    gameName = GameConfig.game_name
    name = Label(root, text=gameName)
    name.config(font=("Courier", 20))
    name.pack()
    link = Label(root, text=TextLoading.getTextData(TextLoading.systemTextPath, 'About')['GoHome'], fg="blue")
    link.pack()
    link.bind("<Button-1>", gohome)
    LICENSE = Label(root, text=TextLoading.getTextData(TextLoading.systemTextPath, 'About')['Licenses'], fg="blue")
    LICENSE.pack()
    LICENSE.bind("<Button-1>", golicense)

    root.mainloop()
