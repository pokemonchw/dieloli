from script.Core import CacheContorl

def panelStateChange(panelId):
    '''
    改变面板状态，若该面板当前状态为0，则更改为1，或者反过来
    Keyword arguments:
    panelId -- 面板id
    '''
    cachePanelState = CacheContorl.panelState[panelId]
    if cachePanelState == '0':
        CacheContorl.panelState[panelId] = "1"
    else:
        CacheContorl.panelState[panelId] = "0"
