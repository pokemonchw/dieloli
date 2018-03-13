import core.CacheContorl as cache

# 改变面板状态
def panelStateChange(panelId):
    cachePanelState = cache.panelState[panelId]
    if cachePanelState == '0':
        cache.panelState[panelId] = "1"
    else:
        cache.panelState[panelId] = "0"
    pass