import core.GameConfig as config
import script.Panel.SaveHandleFramePanel as savehandleframepanel
import core.CacheContorl as cache

# 绘制保存存档页面流程
def establishSave_func():
    inputS = []
    savePage = savePageIndex()
    showSaveValue = savePage[0]
    lastSavePageValue = savePage[1]
    savehandleframepanel.establishSaveInfoHeadPanel()
    flowReturn = savehandleframepanel.seeSaveListPanel(showSaveValue,lastSavePageValue)
    inputS = inputS + flowReturn
    startId = len(inputS)
    flowReturn = savehandleframepanel.askForChangeSavePagePanel(startId)
    inputS = inputS + flowReturn
    pass

# 绘制读取存档页面流程
def loadSave_func():
    inputS = []
    savePage = savePageIndex()
    showSaveValue = savePage[0]
    lastSavePageValue = savePage[1]
    savehandleframepanel.loadSaveInfoHeadPanel()
    flowReturn = savehandleframepanel.seeSaveListPanel(showSaveValue, lastSavePageValue,True)
    inputS = inputS + flowReturn
    startId = len(inputS)
    flowReturn = savehandleframepanel.askForChangeSavePagePanel(startId)
    inputS = inputS + flowReturn
    pass

# 存档页计算
def savePageIndex():
    maxSaveValue = int(config.max_save)
    pageSaveValue = int(config.save_page)
    lastSavePageValue = 0
    if maxSaveValue % pageSaveValue != 0:
        showSaveValue = int(maxSaveValue / pageSaveValue)
        lastSavePageValue = maxSaveValue % pageSaveValue
        cache.maxSavePage = pageSaveValue + 1
    else:
        showSaveValue = maxSaveValue / pageSaveValue
    savePage = [showSaveValue,lastSavePageValue]
    return savePage
