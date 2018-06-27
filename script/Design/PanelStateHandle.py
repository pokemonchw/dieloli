from script.Core import CacheContorl

# 改变面板状态
def panelStateChange(panelId):
    cachePanelState = CacheContorl.panelState[panelId]
    if cachePanelState == '0':
        CacheContorl.panelState[panelId] = "1"
    else:
        CacheContorl.panelState[panelId] = "0"
    pass