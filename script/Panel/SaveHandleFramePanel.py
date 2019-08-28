from script.Core import CacheContorl,GameConfig,PyCmd,TextLoading,EraPrint,TextHandle,SaveHandle
from script.Design import CmdButtonQueue,GameTime

def loadSaveInfoHeadPanel():
    '''
    载入存档信息头面板
    '''
    saveFrameTitle = TextLoading.getTextData(TextLoading.stageWordPath, '71')
    EraPrint.plt(saveFrameTitle)

def establishSaveInfoHeadPanel():
    '''
    存储存档信息头面板
    '''
    saveFrameTitle = TextLoading.getTextData(TextLoading.stageWordPath, '70')
    EraPrint.plt(saveFrameTitle)

def seeSaveListPanel(pageSaveValue:int,lastSavePageValue:int,autoSave = False) -> list:
    '''
    查看存档页面面板
    Keyword arguments:
    pageSaveValue -- 单页最大存档显示数量
    lastSavePageValue -- 最后一页存档显示数量
    autoSave -- 自动存档显示开关 (default False)
    '''
    savePanelPage = int(CacheContorl.panelState['SeeSaveListPanel']) + 1
    inputS = []
    idTextList = []
    idInfoText = TextLoading.getTextData(TextLoading.stageWordPath,'72')
    textWidth = int(GameConfig.text_width)
    saveNoneText = TextLoading.getTextData(TextLoading.messagePath,'20')
    if savePanelPage == int(GameConfig.save_page) + 1:
        startSaveId = int(pageSaveValue) * (savePanelPage - 1)
        overSaveId = startSaveId + lastSavePageValue
    else:
        overSaveId = int(pageSaveValue) * savePanelPage
        startSaveId = overSaveId - int(pageSaveValue)
    for i in range(0,overSaveId - startSaveId):
        id = CmdButtonQueue.idIndex(i)
        saveId = startSaveId + i
        if autoSave == True and SaveHandle.judgeSaveFileExist(saveId) != '1':
            idText = idInfoText + " " + str(saveId) + ":"
            idTextList.append(idText)
        else:
            idText = id + idInfoText + " " + str(saveId) + ":"
            idTextList.append(idText)
    for i in range(0,overSaveId - startSaveId):
        id = str(i)
        idText = idTextList[i]
        EraPrint.plittleline()
        saveId = SaveHandle.getSavePageSaveId(pageSaveValue,i)
        if SaveHandle.judgeSaveFileExist(saveId) == '1':
            saveInfoHead = SaveHandle.loadSaveInfoHead(saveId)
            gameTimeData = saveInfoHead['gameTime']
            gameTimeText = GameTime.getDateText(gameTimeData)
            characterName = saveInfoHead['characterName']
            saveVerson = saveInfoHead['gameVerson']
            saveText = characterName + ' ' + gameTimeText + ' ' + saveVerson
            idTextIndex = int(TextHandle.getTextIndex(idText))
            fixIdWidth = textWidth - idTextIndex
            saveAlign = TextHandle.align(saveText,'center',textWidth=fixIdWidth)
            idText = idText + saveAlign
            PyCmd.pcmd(idText, id, None)
            EraPrint.p('\n')
            inputS.append(id)
        else:
            idTextIndex = int(TextHandle.getTextIndex(idText))
            fixIdWidth = textWidth - idTextIndex
            saveNoneAlign = TextHandle.align(saveNoneText,'center',textWidth=fixIdWidth)
            idText = idText + saveNoneAlign
            if autoSave == True:
                EraPrint.p(idText)
                EraPrint.p('\n')
            else:
                PyCmd.pcmd(idText, id, None)
                inputS.append(id)
                EraPrint.p('\n')
    if autoSave == True:
        autoInfoText = TextLoading.getTextData(TextLoading.stageWordPath,"73")
        i = pageSaveValue
        id = CmdButtonQueue.idIndex(i)
        EraPrint.plittleline()
        if SaveHandle.judgeSaveFileExist('auto') == '1':
            saveInfoHead = SaveHandle.loadSaveInfoHead('auto')
            gameTimeData = saveInfoHead['gameTime']
            gameTimeText = GameTime.getDateText(gameTimeData)
            characterName = saveInfoHead['characterName']
            saveVerson = saveInfoHead['gameVerson']
            saveText = characterName + ' ' + gameTimeText + ' ' + saveVerson
            idText = id + autoInfoText
            idTextIndex = int(TextHandle.getTextIndex(idText))
            fixIdWidth = textWidth - idTextIndex
            saveTextAlign = TextHandle.align(saveText, 'center', textWidth=fixIdWidth)
            idText = idText + saveTextAlign
            PyCmd.pcmd(idText, id, None)
            inputS.append(id)
            EraPrint.p('\n')
        else:
            idTextIndex = int(TextHandle.getTextIndex(autoInfoText))
            fixIdWidth = textWidth - idTextIndex
            saveNoneAlign = TextHandle.align(saveNoneText, 'center', textWidth=fixIdWidth)
            idText = autoInfoText + saveNoneAlign
            EraPrint.p(idText)
            EraPrint.p('\n')
    return inputS

def askForChangeSavePagePanel(startId:str) -> list:
    '''
    询问切换存档页面面板
    Keyword arguments:
    startId -- 面板命令的起始id
    '''
    cmdList = TextLoading.getTextData(TextLoading.cmdPath,"changeSavePage")
    savePanelPage = str(CacheContorl.panelState['SeeSaveListPanel'])
    maxSavePanelPage = str(CacheContorl.maxSavePage)
    savePageText = '(' + savePanelPage + '/' + maxSavePanelPage + ')'
    EraPrint.printPageLine(sample='-',string=savePageText)
    EraPrint.p('\n')
    yrn = CmdButtonQueue.optionint(None, 3, askfor=False, cmdSize='center', startId=startId, cmdListData=cmdList)
    return yrn

def askForOverlaySavePanel() -> list:
    '''
    询问覆盖存档面板
    '''
    EraPrint.p('\n')
    cmdList = TextLoading.getTextData(TextLoading.cmdPath,"overlaySave")
    messageText = TextLoading.getTextData(TextLoading.messagePath,'21')
    EraPrint.pline()
    EraPrint.p(messageText)
    EraPrint.p('\n')
    yrn = CmdButtonQueue.optionint(None, askfor=False, cmdListData=cmdList)
    return yrn

def confirmationOverlaySavePanel() -> list:
    '''
    确认覆盖存档面板
    '''
    EraPrint.p('\n')
    cmdList = TextLoading.getTextData(TextLoading.cmdPath, "confirmationOverlaySave")
    messageText = TextLoading.getTextData(TextLoading.messagePath, '22')
    EraPrint.pline()
    EraPrint.p(messageText)
    EraPrint.p('\n')
    yrn = CmdButtonQueue.optionint(None, askfor=False, cmdListData=cmdList)
    return yrn

def askLoadSavePanel() -> list:
    '''
    询问读取存档面板
    '''
    EraPrint.p('\n')
    cmdList = TextLoading.getTextData(TextLoading.cmdPath,"loadSaveAsk")
    messageText = TextLoading.getTextData(TextLoading.messagePath,'23')
    EraPrint.pline()
    EraPrint.p(messageText)
    EraPrint.p('\n')
    yrn = CmdButtonQueue.optionint(None, askfor=False, cmdListData=cmdList)
    return yrn

def confirmationLoadSavePanel() -> list:
    '''
    确认读取存档面板
    '''
    EraPrint.p('\n')
    cmdList = TextLoading.getTextData(TextLoading.cmdPath, "confirmationLoadSave")
    messageText = TextLoading.getTextData(TextLoading.messagePath, '24')
    EraPrint.pline()
    EraPrint.p(messageText)
    EraPrint.p('\n')
    yrn = CmdButtonQueue.optionint(None, askfor=False, cmdListData=cmdList)
    return yrn

def confirmationRemoveSavePanel() -> list:
    '''
    确认删除存档面板
    '''
    EraPrint.p('\n')
    cmdList = TextLoading.getTextData(TextLoading.cmdPath,"confirmationRemoveSave")
    messageText = TextLoading.getTextData(TextLoading.messagePath, '25')
    EraPrint.pline()
    EraPrint.p(messageText)
    EraPrint.p('\n')
    yrn = CmdButtonQueue.optionint(None, askfor=False, cmdListData=cmdList)
    return yrn
