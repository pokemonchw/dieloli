from script.Core import GameConfig,CacheContorl,GameInit,PyCmd,SaveHandle
from script.Panel import SaveHandleFramePanel

# 绘制保存存档页面流程
def establishSave_func(oldPanel):
    inputS = []
    savePage = savePageIndex()
    showSaveValue = savePage[0]
    lastSavePageValue = savePage[1]
    SaveHandleFramePanel.establishSaveInfoHeadPanel()
    flowReturn = SaveHandleFramePanel.seeSaveListPanel(showSaveValue,lastSavePageValue)
    inputS = inputS + flowReturn
    startId = len(inputS)
    flowReturn = SaveHandleFramePanel.askForChangeSavePagePanel(startId)
    inputS = inputS + flowReturn
    yrn = GameInit.askfor_Int(inputS)
    PyCmd.clr_cmd()
    if yrn == str(startId):
        savePanelPage = int(CacheContorl.panelState['SeeSaveListPanel'])
        if savePanelPage == 0:
            CacheContorl.panelState['SeeSaveListPanel'] = CacheContorl.maxSavePage
        else:
            CacheContorl.panelState['SeeSaveListPanel'] = savePanelPage - 1
        establishSave_func(oldPanel)
    elif yrn == str(startId + 1):
        if oldPanel == 'MainFramePanel':
            CacheContorl.panelState['SeeSaveListPanel'] = 0
            import script.Flow.Main as mainframe
            mainframe.mainFrame_func()
            pass
        else:
            pass
    elif yrn == str(startId + 2):
        savePanelPage = int(CacheContorl.panelState['SeeSaveListPanel'])
        if savePanelPage == CacheContorl.maxSavePage:
            CacheContorl.panelState['SeeSaveListPanel'] = 0
        else:
            CacheContorl.panelState['SeeSaveListPanel'] = savePanelPage + 1
        establishSave_func(oldPanel)
    else:
        ansReturn = int(yrn)
        saveId = SaveHandle.getSavePageSaveId(showSaveValue,ansReturn)
        if SaveHandle.judgeSaveFileExist(saveId) == '1':
            askForOverlaySave_func(oldPanel,saveId)
        else:
            SaveHandle.establishSave(saveId)
            establishSave_func(oldPanel)
    pass

# 绘制读取存档页面流程
def loadSave_func(oldPanel):
    inputS = []
    savePage = savePageIndex()
    showSaveValue = savePage[0]
    lastSavePageValue = savePage[1]
    SaveHandleFramePanel.loadSaveInfoHeadPanel()
    flowReturn = SaveHandleFramePanel.seeSaveListPanel(showSaveValue, lastSavePageValue,True)
    inputS = inputS + flowReturn
    startId = len(inputS)
    flowReturn = SaveHandleFramePanel.askForChangeSavePagePanel(startId)
    inputS = inputS + flowReturn
    yrn = GameInit.askfor_Int(inputS)
    PyCmd.clr_cmd()
    if yrn == str(startId):
        savePanelPage = int(CacheContorl.panelState['SeeSaveListPanel'])
        if savePanelPage == 0:
            CacheContorl.panelState['SeeSaveListPanel'] = CacheContorl.maxSavePage
        else:
            CacheContorl.panelState['SeeSaveListPanel'] = savePanelPage - 1
        loadSave_func(oldPanel)
    elif yrn == str(startId + 1):
        if oldPanel == 'MainFlowPanel':
            CacheContorl.wframeMouse['wFrameRePrint'] = 1
            CacheContorl.panelState['SeeSaveListPanel'] = 0
            import script.Design.StartFlow as mainflow
            mainflow.main_func()
        elif oldPanel == 'MainFramePanel':
            CacheContorl.panelState['SeeSaveListPanel'] = 0
            import script.Flow.Main as mainframe
            mainframe.mainFrame_func()
            pass
    elif yrn == str(startId + 2):
        savePanelPage = int(CacheContorl.panelState['SeeSaveListPanel'])
        if savePanelPage == CacheContorl.maxSavePage:
            CacheContorl.panelState['SeeSaveListPanel'] = 0
        else:
            CacheContorl.panelState['SeeSaveListPanel'] = savePanelPage + 1
        loadSave_func(oldPanel)
    else:
        ansReturn = int(yrn)
        saveId = SaveHandle.getSavePageSaveId(showSaveValue,ansReturn)
        askForLoadSave_func(oldPanel,saveId)
    pass

# 存档页计算
def savePageIndex():
    maxSaveValue = int(GameConfig.max_save)
    pageSaveValue = int(GameConfig.save_page)
    lastSavePageValue = 0
    if maxSaveValue % pageSaveValue != 0:
        showSaveValue = int(maxSaveValue / pageSaveValue)
        lastSavePageValue = maxSaveValue % pageSaveValue
        CacheContorl.maxSavePage = pageSaveValue
    else:
        CacheContorl.maxSavePage = pageSaveValue - 1
        showSaveValue = maxSaveValue / pageSaveValue
    savePage = [showSaveValue,lastSavePageValue]
    return savePage

# 询问覆盖存档流程
def askForOverlaySave_func(oldPanel,saveId):
    cmdList = SaveHandleFramePanel.askForOverlaySavePanel()
    yrn = GameInit.askfor_All(cmdList)
    yrn = str(yrn)
    PyCmd.clr_cmd()
    if yrn == '0':
        confirmationOverlaySave_func(oldPanel,saveId)
    elif yrn == '1':
        confirmationRemoveSave_func(saveId,'EstablishSavePanel',oldPanel)
    elif yrn == '2':
        establishSave_func(oldPanel)
    pass

# 确认覆盖流程
def confirmationOverlaySave_func(oldPanel,saveId):
    cmdList = SaveHandleFramePanel.confirmationOverlaySavePanel()
    yrn = GameInit.askfor_All(cmdList)
    PyCmd.clr_cmd()
    if yrn == '0':
        SaveHandle.establishSave(saveId)
        establishSave_func(oldPanel)
    else:
        establishSave_func(oldPanel)
    pass

# 询问读取存档流程
def askForLoadSave_func(oldPanel,saveId):
    cmdList = SaveHandleFramePanel.askLoadSavePanel()
    yrn = GameInit.askfor_All(cmdList)
    PyCmd.clr_cmd()
    if yrn == '0':
        confirmationLoadSave_func(oldPanel,saveId)
    elif yrn == '1':
        confirmationRemoveSave_func(saveId,'LoadSavePanel',oldPanel)
    elif yrn == '2':
        loadSave_func(oldPanel)
    pass

# 确认读取存档流程
def confirmationLoadSave_func(oldPanel,saveId):
    cmdList = SaveHandleFramePanel.confirmationLoadSavePanel()
    yrn = GameInit.askfor_All(cmdList)
    PyCmd.clr_cmd()
    if yrn == '0':
        SaveHandle.inputLoadSave(saveId)
        import script.Flow.Main as mainframe
        mainframe.mainFrame_func()
    else:
        loadSave_func(oldPanel)
    pass

# 确认删除存档流程
def confirmationRemoveSave_func(saveId,oldPanel,tooOldPanel):
    cmdList = SaveHandleFramePanel.confirmationRemoveSavePanel()
    yrn = GameInit.askfor_All(cmdList)
    if yrn == '0':
        SaveHandle.removeSave(saveId)
    else:
        pass
    PyCmd.clr_cmd()
    if oldPanel == 'LoadSavePanel':
        loadSave_func(tooOldPanel)
    elif oldPanel == 'EstablishSavePanel':
        establishSave_func(tooOldPanel)
    pass
