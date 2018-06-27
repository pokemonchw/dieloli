from script.Core import MainFrame,PyCmd,CacheContorl

wframe = MainFrame.root

#按键监听绑定预设
def onWFrameListion():
    wframe.bind('<ButtonPress-1>', mouseLeftCheck)
    wframe.bind('<ButtonPress-3>',mouseRightCheck)
    wframe.bind('<Return>', MainFrame.send_input)
    wframe.bind('<KP_Enter>', MainFrame.send_input)
    wframe.bind('<Up>',keyUp)
    wframe.bind('<Down>',keyDown)

#鼠标左键事件
def mouseLeftCheck(event):
    if CacheContorl.wframeMouse['wFrameUp'] ==0:
        setWFrameUp(event)
    else:
        mouseCheckPush(event)

#鼠标右键事件
def mouseRightCheck(event):
    CacheContorl.wframeMouse['mouseRight'] = 1
    CacheContorl.textWait = 0
    if CacheContorl.wframeMouse['wFrameUp'] ==0:
        setWFrameUp(event)
    else:
        mouseCheckPush(event)

#键盘上键事件
def keyUp(event):
    while CacheContorl.inputPosition['position'] == 0:
        CacheContorl.inputPosition['position'] = len(CacheContorl.inputCache)
    while CacheContorl.inputPosition['position'] <= 21 and CacheContorl.inputPosition['position'] > 1:
        CacheContorl.inputPosition['position'] = CacheContorl.inputPosition['position'] - 1
        inpotId = CacheContorl.inputPosition['position']
        try:
            MainFrame.order.set(CacheContorl.inputCache[inpotId])
            break
        except KeyError:
            CacheContorl.inputPosition['position'] = CacheContorl.inputPosition['position'] + 1
    pass

#键盘下键事件
def keyDown(event):
    if CacheContorl.inputPosition['position'] > 0 and CacheContorl.inputPosition['position'] < len(CacheContorl.inputCache) - 1:
        try:
            CacheContorl.inputPosition['position'] = CacheContorl.inputPosition['position'] + 1
            inpotId = CacheContorl.inputPosition['position']
            MainFrame.order.set(CacheContorl.inputCache[inpotId])
        except KeyError:
            CacheContorl.inputPosition['position'] = CacheContorl.inputPosition['position'] - 1
    elif CacheContorl.inputPosition['position'] == len(CacheContorl.inputCache) - 1:
        CacheContorl.inputPosition['position'] = 0
        MainFrame.order.set('')
    pass

#逐字输出状态改变
def setWFrameUp(event):
    CacheContorl.wframeMouse['wFrameUp'] = 1
    CacheContorl.wframeMouse['wFrameLinesUp'] = 1

#鼠标点击状态
def mouseCheckPush(event):
    PyCmd.focusCmd()
    if CacheContorl.wframeMouse['mouseLeaveCmd'] == 0:
        MainFrame.send_input()
        CacheContorl.wframeMouse['mouseLeaveCmd'] = 1
    else:
        pass
