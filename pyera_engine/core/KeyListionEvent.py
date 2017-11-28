import core.winframe as winframe
import core.PyCmd as pycmd

wframe = winframe.root

wframeMouse = {'wFrameUp':2}

def onWFrameMouse():
    wframe.bind('<ButtonPress-1>', mouseLeftCheck)

def mouseLeftCheck(event):
    if wframeMouse['wFrameUp'] ==0:
        setWFrameUp(event)
    else:
        pycmd.focusCmd()

def setWFrameUp(event):
    wframeMouse['wFrameUp'] = 1