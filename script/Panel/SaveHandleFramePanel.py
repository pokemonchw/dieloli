import core.CacheContorl as cache
import core.GameConfig as config
import core.PyCmd as pycmd

# 查看存档页面面板
def seeSaveListPanel(pageSaveValue,lastSavePageValue):
    savePanelPage = int(cache.panelState['SeeSaveListPanel']) + 1
    inputS = []
    print(pageSaveValue)
    if savePanelPage == int(config.save_page) + 1:
        startSaveId = int(pageSaveValue) * savePanelPage + 1
        overSaveId = startSaveId + lastSavePageValue
    else:
        overSaveId = int(pageSaveValue) * savePanelPage
        startSaveId = overSaveId - int(pageSaveValue)
    for i in range(startSaveId,overSaveId):
        id = str(i)
        pycmd.pcmd(id,id,None)
        inputS.append(id)
    pass