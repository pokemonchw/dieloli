from script.Core import TextLoading,GameConfig
from tkinter import Label,Tk
import webbrowser


def gohome(event):
    webbrowser.open(r"https://github.com/pokemonchw/dieloli")

def golicense(event):
    webbrowser.open(r"http://creativecommons.org/licenses/by-nc-sa/2.0/")

# 设置菜单
def openAboutFrame():
    root = Tk()
    titleText = TextLoading.getTextData(TextLoading.systemId, 'About')['TitleName']
    root.title(titleText)
    gameName = GameConfig.game_name
    name = Label(root, text=gameName)
    name.config(font=("Courier", 20))
    name.pack()
    link = Label(root, text=TextLoading.getTextData(TextLoading.systemId, 'About')['GoHome'], fg="blue")
    link.pack()
    link.bind("<Button-1>", gohome)
    LICENSE = Label(root, text=TextLoading.getTextData(TextLoading.systemId, 'About')['Licenses'], fg="blue")
    LICENSE.pack()
    LICENSE.bind("<Button-1>", golicense)

    root.mainloop()
