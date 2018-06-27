from script.Core import CacheContorl,GameConfig,PyCmd,TextLoading,EraPrint,TextHandle,SaveHandle
from script.Design import CmdButtonQueue,GameTime

# 载入存档信息头面板
def loadSaveInfoHeadPanel():
    saveFrameTitle = TextLoading.getTextData(TextLoading.stageWordId, '71')
    EraPrint.plt(saveFrameTitle)
    pass

# 储存存档信息头面板
def establishSaveInfoHeadPanel():
    saveFrameTitle = TextLoading.getTextData(TextLoading.stageWordId, '70')
    EraPrint.plt(saveFrameTitle)
    pass

# 查看存档页面面板
def seeSaveListPanel(pageSaveValue,lastSavePageValue,autoSave = False):
    savePanelPage = int(CacheContorl.panelState['SeeSaveListPanel']) + 1
    inputS = []
    idTextList = []
    idInfoText = TextLoading.getTextData(TextLoading.stageWordId,'72')
    textWidth = int(GameConfig.text_width)
    saveNoneText = TextLoading.getTextData(TextLoading.messageId,'20')
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
        saveid = SaveHandle.getSavePageSaveId(pageSaveValue,i)
        if SaveHandle.judgeSaveFileExist(saveid) == '1':
            saveData = SaveHandle.loadSave(saveid)
            playerData = saveData['playerData']
            gameTimeData = saveData['gameTime']
            gameTimeText = GameTime.getDateText(gameTimeData)
            playerName = playerData['object']['0']['Name']
            saveVerson = saveData['gameVerson']
            saveText = playerName + ' ' + gameTimeText + ' ' + saveVerson
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
        autoInfoText = TextLoading.getTextData(TextLoading.stageWordId,"73")
        i = pageSaveValue
        id = CmdButtonQueue.idIndex(i)
        EraPrint.plittleline()
        if SaveHandle.judgeSaveFileExist('auto') == '1':
            saveData = SaveHandle.loadSave('auto')
            playerData = saveData['playerData']
            gameTimeData = saveData['gameTime']
            gameTimeText = GameTime.getDateText(gameTimeData)
            saveVerson = saveData['gameVerson']
            playerName = playerData['object']['0']['Name']
            saveText = playerName + ' ' + gameTimeText + ' ' + saveVerson
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
    else:
        pass
    return inputS

# 询问切换存档页面板
def askForChangeSavePagePanel(startId):
    cmdList = TextLoading.getTextData(TextLoading.cmdId,"changeSavePage")
    savePanelPage = str(CacheContorl.panelState['SeeSaveListPanel'])
    maxSavePanelPage = str(CacheContorl.maxSavePage)
    savePageText = '(' + savePanelPage + '/' + maxSavePanelPage + ')'
    EraPrint.printPageLine(sample='-',string=savePageText)
    EraPrint.p('\n')
    yrn = CmdButtonQueue.optionint(None, 3, askfor=False, cmdSize='center', startId=startId, cmdListData=cmdList)
    return yrn

# 询问覆盖存档面板
def askForOverlaySavePanel():
    EraPrint.p('\n')
    cmdList = TextLoading.getTextData(TextLoading.cmdId,"overlaySave")
    messageText = TextLoading.getTextData(TextLoading.messageId,'21')
    EraPrint.pline()
    EraPrint.p(messageText)
    EraPrint.p('\n')
    yrn = CmdButtonQueue.optionint(None, 1, askfor=False, cmdListData=cmdList)
    return yrn

# 确认覆盖面板
def confirmationOverlaySavePanel():
    EraPrint.p('\n')
    cmdList = TextLoading.getTextData(TextLoading.cmdId, "confirmationOverlaySave")
    messageText = TextLoading.getTextData(TextLoading.messageId, '22')
    EraPrint.pline()
    EraPrint.p(messageText)
    EraPrint.p('\n')
    yrn = CmdButtonQueue.optionint(None, 1, askfor=False, cmdListData=cmdList)
    return yrn

# 询问读档面板
def askLoadSavePanel():
    EraPrint.p('\n')
    cmdList = TextLoading.getTextData(TextLoading.cmdId,"loadSaveAsk")
    messageText = TextLoading.getTextData(TextLoading.messageId,'23')
    EraPrint.pline()
    EraPrint.p(messageText)
    EraPrint.p('\n')
    yrn = CmdButtonQueue.optionint(None, 1, askfor=False, cmdListData=cmdList)
    return yrn

# 确认读档面板
def confirmationLoadSavePanel():
    EraPrint.p('\n')
    cmdList = TextLoading.getTextData(TextLoading.cmdId, "confirmationLoadSave")
    messageText = TextLoading.getTextData(TextLoading.messageId, '24')
    EraPrint.pline()
    EraPrint.p(messageText)
    EraPrint.p('\n')
    yrn = CmdButtonQueue.optionint(None, 1, askfor=False, cmdListData=cmdList)
    return yrn

# 确认删除存档面板
def confirmationRemoveSavePanel():
    EraPrint.p('\n')
    cmdList = TextLoading.getTextData(TextLoading.cmdId,"confirmationRemoveSave")
    messageText = TextLoading.getTextData(TextLoading.messageId, '25')
    EraPrint.pline()
    EraPrint.p(messageText)
    EraPrint.p('\n')
    yrn = CmdButtonQueue.optionint(None, 1, askfor=False, cmdListData=cmdList)
    return yrn