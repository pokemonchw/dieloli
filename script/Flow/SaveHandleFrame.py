from script.Core import GameConfig,CacheContorl,GameInit,PyCmd,SaveHandle
from script.Panel import SaveHandleFramePanel

# 绘制保存存档页面流程
def establishSave_func():
    while(True):
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
        elif yrn == str(startId + 1):
            CacheContorl.panelState['SeeSaveListPanel'] = 0
            CacheContorl.nowFlowId = CacheContorl.oldFlowId
            break
        elif yrn == str(startId + 2):
            savePanelPage = int(CacheContorl.panelState['SeeSaveListPanel'])
            if savePanelPage == CacheContorl.maxSavePage:
                CacheContorl.panelState['SeeSaveListPanel'] = 0
            else:
                CacheContorl.panelState['SeeSaveListPanel'] = savePanelPage + 1
        else:
            ansReturn = int(yrn)
            saveId = SaveHandle.getSavePageSaveId(showSaveValue,ansReturn)
            if SaveHandle.judgeSaveFileExist(saveId) == '1':
                askForOverlaySave_func(saveId)
            else:
                SaveHandle.establishSave(saveId)

# 绘制读取存档页面流程
def loadSave_func():
    while(True):
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
        elif yrn == str(startId + 1):
            CacheContorl.panelState['SeeSaveListPanel'] = 0
            CacheContorl.nowFlowId = CacheContorl.oldFlowId
            break
        elif yrn == str(startId + 2):
            savePanelPage = int(CacheContorl.panelState['SeeSaveListPanel'])
            if savePanelPage == CacheContorl.maxSavePage:
                CacheContorl.panelState['SeeSaveListPanel'] = 0
            else:
                CacheContorl.panelState['SeeSaveListPanel'] = savePanelPage + 1
        else:
            ansReturn = int(yrn)
            saveId = SaveHandle.getSavePageSaveId(showSaveValue,ansReturn)
            if askForLoadSave_func(saveId):
                break

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
def askForOverlaySave_func(saveId):
    cmdList = SaveHandleFramePanel.askForOverlaySavePanel()
    yrn = GameInit.askfor_All(cmdList)
    yrn = str(yrn)
    PyCmd.clr_cmd()
    if yrn == '0':
        confirmationOverlaySave_func(saveId)
    elif yrn == '1':
        confirmationRemoveSave_func(saveId)

# 确认覆盖流程
def confirmationOverlaySave_func(saveId):
    cmdList = SaveHandleFramePanel.confirmationOverlaySavePanel()
    yrn = GameInit.askfor_All(cmdList)
    PyCmd.clr_cmd()
    if yrn == '0':
        SaveHandle.establishSave(saveId)
    return

# 询问读取存档流程
def askForLoadSave_func(saveId):
    cmdList = SaveHandleFramePanel.askLoadSavePanel()
    yrn = GameInit.askfor_All(cmdList)
    PyCmd.clr_cmd()
    returnJudge = False
    if yrn == '0':
        returnJudge = True
        confirmationLoadSave_func(saveId)
    elif yrn == '1':
        returnJudge = True
        confirmationRemoveSave_func(saveId)
    return returnJudge

# 确认读取存档流程
def confirmationLoadSave_func(saveId):
    cmdList = SaveHandleFramePanel.confirmationLoadSavePanel()
    yrn = GameInit.askfor_All(cmdList)
    PyCmd.clr_cmd()
    if yrn == '0':
        SaveHandle.inputLoadSave(saveId)
        CacheContorl.nowFlowId = 'main'

# 确认删除存档流程
def confirmationRemoveSave_func(saveId):
    cmdList = SaveHandleFramePanel.confirmationRemoveSavePanel()
    yrn = GameInit.askfor_All(cmdList)
    if yrn == '0':
        SaveHandle.removeSave(saveId)
    PyCmd.clr_cmd()
