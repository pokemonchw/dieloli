import core.winframe as winframe

wframe = winframe.root

wframeMouse = {}

def onWFrameMouse():
    wframeMouse["leftMouse"] = 0
    wframe.bind('<ButtonPress-1>',setWFrameUp)

def offWFrameMouse():
    wframeMouse["leftMouse"] = 0
    wframe.unbind('<ButtonPress-1>')

def setWFrameUp(event):
    wframeMouse["leftMouse"] = 1