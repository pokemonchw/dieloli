import core.winframe as winframe
import core.PyCmd as pycmd
import core.CacheContorl as cache

wframe = winframe.root

def onWFrameMouse():
    wframe.bind('<ButtonPress-1>', mouseLeftCheck)
    wframe.bind('<ButtonPress-3>',mouseRightCheck)
    wframe.bind('<Return>', winframe.send_input)

def mouseLeftCheck(event):
    if cache.wframeMouse['wFrameUp'] ==0:
        setWFrameUp(event)
    else:
        mouseCheckPush(event)

def mouseRightCheck(event):
    cache.wframeMouse['mouseRight'] = 1

def setWFrameUp(event):
    cache.wframeMouse['wFrameUp'] = 1

def mouseCheckPush(event):
    pycmd.focusCmd()
    if cache.wframeMouse['mouseLeaveCmd'] == 0:
        winframe.send_input()
        cache.wframeMouse['mouseLeaveCmd'] = 1
    else:
        pass
