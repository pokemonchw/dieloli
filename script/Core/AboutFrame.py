from script.Core import TextLoading, GameConfig
from tkinter import Label, Tk, Event
import webbrowser

def gohome(event:Event):
    '''
    通过浏览器打开到游戏主页的链接
    Keyword arguments:
    event -- 点击事件
    '''
    webbrowser.open(r"https://github.com/pokemonchw/dieloli")

def golicense(event:Event):
    '''
    通过浏览器打开协议链接
    keyword arguments:
    event -- 点击事件
    '''
    webbrowser.open(r"http://creativecommons.org/licenses/by-nc-sa/2.0/")

def openAboutFrame():
    '''
    打开设置菜单
    '''
    root = Tk()
    title_text = TextLoading.getTextData(TextLoading.systemTextPath, 'About')['TitleName']
    root.title(title_text)
    gameName = GameConfig.game_name
    name = Label(root, text=gameName)
    name.config(font=("Courier", 20))
    name.pack()
    linkInfo = TextLoading.getTextData(TextLoading.systemTextPath,'About')['GoHome']
    link = Label(root,text=linkInfo,fg="blue")
    link.pack()
    link.bind("<Button-1>", gohome)
    licenseInfo = TextLoading.getTextData(TextLoading.systemTextPath, 'About')['Licenses']
    LICENSE = Label(root,text=licenseInfo,fg="blue")
    LICENSE.pack()
    LICENSE.bind("<Button-1>", golicense)

    root.mainloop()
