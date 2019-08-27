from script.Core import GameConfig,CacheContorl,GameInit,PyCmd,SaveHandle
from script.Panel import SaveHandleFramePanel

def establishSave_func():
    '''
    绘制保存存档界面流程
    '''
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

def loadSave_func():
    '''
    绘制读取存档界面流程
    '''
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

def savePageIndex():
    '''
    用于计算存档页面单页存档显示数量
    return:
    savePage[0] -- 存档页面单页显示数量
    savePage[1] -- 最大存档数不能被存档页数整除时，额外存档页存档数量
    '''
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

def askForOverlaySave_func(saveId):
    '''
    存档处理询问流程
    玩家输入0:进入覆盖存档询问流程
    玩家输入1:进入删除存档询问流程
    Keyword arguments:
    saveId -- 存档id
    '''
    cmdList = SaveHandleFramePanel.askForOverlaySavePanel()
    yrn = GameInit.askfor_All(cmdList)
    yrn = str(yrn)
    PyCmd.clr_cmd()
    if yrn == '0':
        confirmationOverlaySave_func(saveId)
    elif yrn == '1':
        confirmationRemoveSave_func(saveId)

def confirmationOverlaySave_func(saveId):
    '''
    覆盖存档询问流程
    玩家输入0:对存档进行覆盖
    Keyword arguments:
    saveId -- 存档id
    '''
    cmdList = SaveHandleFramePanel.confirmationOverlaySavePanel()
    yrn = GameInit.askfor_All(cmdList)
    PyCmd.clr_cmd()
    if yrn == '0':
        SaveHandle.establishSave(saveId)

def askForLoadSave_func(saveId):
    '''
    读档处理询问流程
    玩家输入0:进入读取存档询问流程
    玩家输入1:进入删除存档询问流程
    Keyword arguments:
    saveId -- 存档id
    '''
    cmdList = SaveHandleFramePanel.askLoadSavePanel()
    yrn = GameInit.askfor_All(cmdList)
    PyCmd.clr_cmd()
    if yrn == '0':
        return confirmationLoadSave_func(saveId)
    elif yrn == '1':
        confirmationRemoveSave_func(saveId)
    return False

def confirmationLoadSave_func(saveId):
    '''
    读取存档询问流程
    玩家输入0:读取指定存档
    Keyword arguments:
    saveId -- 存档id
    '''
    cmdList = SaveHandleFramePanel.confirmationLoadSavePanel()
    yrn = GameInit.askfor_All(cmdList)
    PyCmd.clr_cmd()
    if yrn == '0':
        SaveHandle.inputLoadSave(saveId)
        CacheContorl.nowFlowId = 'main'
        return True
    return False

def confirmationRemoveSave_func(saveId):
    '''
    覆盖存档询问流程
    玩家输入0:删除指定存档
    Keyword arguments:
    saveId -- 存档id
    '''
    cmdList = SaveHandleFramePanel.confirmationRemoveSavePanel()
    yrn = GameInit.askfor_All(cmdList)
    if yrn == '0':
        SaveHandle.removeSave(saveId)
    PyCmd.clr_cmd()
