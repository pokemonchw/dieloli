import core.GameConfig as config
import script.Panel.SaveHandleFramePanel as savehandleframepanel
import core.CacheContorl as cache
import core.game as game
import core.PyCmd as pycmd
import core.SaveHandle as savehandle

# 绘制保存存档页面流程
def establishSave_func(oldPanel):
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
    yrn = game.askfor_Int(inputS)
    if yrn == str(startId):
        savePanelPage = int(cache.panelState['SeeSaveListPanel'])
        if savePanelPage == 0:
            cache.panelState['SeeSaveListPanel'] = cache.maxSavePage
        else:
            cache.panelState['SeeSaveListPanel'] = savePanelPage - 1
        pycmd.clr_cmd()
        establishSave_func(oldPanel)
    elif yrn == str(startId + 1):
        if oldPanel == 'MainFramePanel':
            pycmd.clr_cmd()
            import script.flow.MainFrame as mainframe
            mainframe.mainFrame_func()
            pass
        else:
            pass
    elif yrn == str(startId + 2):
        savePanelPage = int(cache.panelState['SeeSaveListPanel'])
        if savePanelPage == cache.maxSavePage:
            cache.panelState['SeeSaveListPanel'] = 0
        else:
            cache.panelState['SeeSaveListPanel'] = savePanelPage + 1
        pycmd.clr_cmd()
        establishSave_func(oldPanel)
    else:
        ansReturn = int(yrn)
        saveId = savehandle.getSavePageSaveId(showSaveValue,ansReturn)
        if savehandle.judgeSaveFileExist(saveId) == '1':
            askForOverlaySave_func(oldPanel,saveId)
        else:
            savehandle.establishSave(saveId)
            establishSave_func(oldPanel)
    pass

# 绘制读取存档页面流程
def loadSave_func(oldPanel):
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
    yrn = game.askfor_Int(inputS)
    if yrn == str(startId):
        savePanelPage = int(cache.panelState['SeeSaveListPanel'])
        if savePanelPage == 0:
            cache.panelState['SeeSaveListPanel'] = cache.maxSavePage
        else:
            cache.panelState['SeeSaveListPanel'] = savePanelPage - 1
        pycmd.clr_cmd()
        loadSave_func(oldPanel)
    elif yrn == str(startId + 1):
        if oldPanel == 'MainFlowPanel':
            pycmd.clr_cmd()
            cache.wframeMouse['wFrameRePrint'] = 1
            import script.mainflow as mainflow
            mainflow.main_func()
        elif oldPanel == 'MainFramePanel':
            pycmd.clr_cmd()
            import script.flow.MainFrame as mainframe
            mainframe.mainFrame_func()
            pass
    elif yrn == str(startId + 2):
        savePanelPage = int(cache.panelState['SeeSaveListPanel'])
        if savePanelPage == cache.maxSavePage:
            cache.panelState['SeeSaveListPanel'] = 0
        else:
            cache.panelState['SeeSaveListPanel'] = savePanelPage + 1
        pycmd.clr_cmd()
        loadSave_func(oldPanel)
    else:
        ansReturn = int(yrn)
        saveId = savehandle.getSavePageSaveId(showSaveValue,ansReturn)
        savehandle.inputLoadSave(saveId)
        import script.flow.MainFrame as mainframe
        mainframe.mainFrame_func()
    pass

# 存档页计算
def savePageIndex():
    maxSaveValue = int(config.max_save)
    pageSaveValue = int(config.save_page)
    lastSavePageValue = 0
    if maxSaveValue % pageSaveValue != 0:
        showSaveValue = int(maxSaveValue / pageSaveValue)
        lastSavePageValue = maxSaveValue % pageSaveValue
        cache.maxSavePage = pageSaveValue
    else:
        cache.maxSavePage = pageSaveValue - 1
        showSaveValue = maxSaveValue / pageSaveValue
    savePage = [showSaveValue,lastSavePageValue]
    return savePage

# 确认覆盖存档流程
def askForOverlaySave_func(oldPanel,saveId):
    cmdList = savehandleframepanel.askForOverlaySavePanel()
    yrn = game.askfor_All(cmdList)
    yrn = str(yrn)
    if yrn == '0':
        savehandle.establishSave(saveId)
        establishSave_func(oldPanel)
    else:
        establishSave_func(oldPanel)
    pass