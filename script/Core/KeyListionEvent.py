from script.Core import MainFrame,PyCmd,CacheContorl

wframe = MainFrame.root

def onWFrameListion():
    '''
    对按键事件进行绑定
    '''
    wframe.bind('<ButtonPress-1>', mouseLeftCheck)
    wframe.bind('<ButtonPress-3>',mouseRightCheck)
    wframe.bind('<Return>', MainFrame.send_input)
    wframe.bind('<KP_Enter>', MainFrame.send_input)
    wframe.bind('<Up>',keyUp)
    wframe.bind('<Down>',keyDown)

def mouseLeftCheck(event):
    '''
    鼠标左键事件处理
    Keyword arguments:
    event -- 鼠标事件
    '''
    if CacheContorl.wframeMouse['wFrameUp'] ==0:
        setWFrameUp()
    else:
        mouseCheckPush()

def mouseRightCheck(event):
    '''
    鼠标右键事件处理
    Keyword arguments:
    event -- 鼠标事件
    '''
    CacheContorl.wframeMouse['mouseRight'] = 1
    CacheContorl.textWait = 0
    if CacheContorl.wframeMouse['wFrameUp'] ==0:
        setWFrameUp()
    else:
        mouseCheckPush()

def keyUp(event):
    '''
    键盘上键事件处理
    Keyword arguments:
    event -- 键盘事件
    '''
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

def keyDown(event):
    '''
    键盘下键事件处理
    Keyword arguments:
    event -- 键盘事件
    '''
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

def setWFrameUp():
    '''
    修正逐字输出状态为nowait
    '''
    CacheContorl.wframeMouse['wFrameUp'] = 1
    CacheContorl.wframeMouse['wFrameLinesUp'] = 1

def mouseCheckPush():
    '''
    更正鼠标点击状态数据映射
    '''
    PyCmd.focusCmd()
    if CacheContorl.wframeMouse['mouseLeaveCmd'] == 0:
        MainFrame.send_input()
        CacheContorl.wframeMouse['mouseLeaveCmd'] = 1
