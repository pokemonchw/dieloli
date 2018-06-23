import Core.TextLoading as textload
import tkinter
import webbrowser


def gohome(event):
    webbrowser.open(r"https://github.com/pokemonchw/dieloli")


def golicense(event):
    webbrowser.open(r"http://creativecommons.org/licenses/by-nc-sa/2.0/")

# 设置菜单
def openAboutFrame():
    root = tkinter.Tk()
    titleText = textload.getTextData(textload.systemId, 'About')['TitleName']
    root.title(titleText)
    root.geometry('150x150')
    name = tkinter.Label(root, text="Deloli")
    name.config(font=("Courier", 20))
    name.place(relwidth=1, relheight=1)
    name.pack()

    link = tkinter.Label(root, text="Go Home", fg="blue")
    link.place(x=120, y=200, relwidth=0.8, relheight=0.4)
    link.pack()
    link.bind("<Button-1>", gohome)

    LICENSE = tkinter.Label(root, text="License: cc by-nc-sa", fg="blue")
    LICENSE.pack()
    LICENSE.bind("<Button-1>", golicense)

    root.mainloop()
