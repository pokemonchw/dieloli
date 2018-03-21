import core.CacheContorl as cache
import core.GameConfig as config
import core.PyCmd as pycmd
import core.TextLoading as textload
import core.EraPrint as eprint
import script.Ans as ans
import core.TextHandle as text
import core.SaveHandle as savehandle
import script.GameTime as gametime

# 载入存档信息头面板
def loadSaveInfoHeadPanel():
    saveFrameTitle = textload.getTextData(textload.stageWordId, '71')
    eprint.plt(saveFrameTitle)
    pass

# 储存存档信息头面板
def establishSaveInfoHeadPanel():
    saveFrameTitle = textload.getTextData(textload.stageWordId, '70')
    eprint.plt(saveFrameTitle)
    pass

# 查看存档页面面板
def seeSaveListPanel(pageSaveValue,lastSavePageValue,autoSave = False):
    savePanelPage = int(cache.panelState['SeeSaveListPanel']) + 1
    inputS = []
    idTextList = []
    idInfoText = textload.getTextData(textload.stageWordId,'72')
    textWidth = int(config.text_width)
    saveNoneText = textload.getTextData(textload.messageId,'20')
    if savePanelPage == int(config.save_page) + 1:
        startSaveId = int(pageSaveValue) * (savePanelPage - 1)
        overSaveId = startSaveId + lastSavePageValue
    else:
        overSaveId = int(pageSaveValue) * savePanelPage
        startSaveId = overSaveId - int(pageSaveValue)
    for i in range(0,overSaveId - startSaveId):
        id = ans.idIndex(i)
        saveId = startSaveId + i
        if autoSave == False:
            idText = id + idInfoText + " " + str(saveId) + ":"
            idTextList.append(idText)
        else:
            if savehandle.judgeSaveFileExist(i) == '1':
                idText = id + idInfoText + " " + str(saveId) + ":"
                idTextList.append(idText)
            else:
                idText = idInfoText + " " + str(saveId) + ":"
                idTextList.append(idText)
    for i in range(0,overSaveId - startSaveId):
        id = str(i)
        idText = idTextList[i]
        eprint.plittleline()
        saveid = savehandle.getSavePageSaveId(pageSaveValue,i)
        if savehandle.judgeSaveFileExist(saveid) == '1':
            saveData = savehandle.loadSave(saveid)
            playerData = saveData['playerData']
            gameTimeData = saveData['gameTime']
            gameTimeText = gametime.getDateText(gameTimeData)
            playerName = playerData['object']['0']['Name']
            saveVerson = saveData['gameVerson']
            saveText = playerName + ' ' + gameTimeText + ' ' + saveVerson
            idTextIndex = int(text.getTextIndex(idText))
            fixIdWidth = textWidth - idTextIndex
            saveAlign = text.align(saveText,'center',textWidth=fixIdWidth)
            idText = idText + saveAlign
            pycmd.pcmd(idText, id, None)
            eprint.p('\n')
            inputS.append(id)
        else:
            idTextIndex = int(text.getTextIndex(idText))
            fixIdWidth = textWidth - idTextIndex
            saveNoneAlign = text.align(saveNoneText,'center',textWidth=fixIdWidth)
            idText = idText + saveNoneAlign
            if autoSave == True:
                eprint.p(idText)
                eprint.p('\n')
            else:
                pycmd.pcmd(idText, id, None)
                inputS.append(id)
                eprint.p('\n')
    if autoSave == True:
        autoInfoText = textload.getTextData(textload.stageWordId,"73")
        i = pageSaveValue
        id = ans.idIndex(i)
        eprint.plittleline()
        if savehandle.judgeSaveFileExist('auto') == '1':
            saveData = savehandle.loadSave('auto')
            playerData = saveData['playerData']
            gameTimeData = saveData['gameTime']
            gameTimeText = gametime.getDateText(gameTimeData)
            saveVerson = saveData['gameVerson']
            playerName = playerData['object']['0']['Name']
            saveText = playerName + ' ' + gameTimeText + ' ' + saveVerson
            idText = id + autoInfoText
            idTextIndex = int(text.getTextIndex(idText))
            fixIdWidth = textWidth - idTextIndex
            saveTextAlign = text.align(saveText, 'center', textWidth=fixIdWidth)
            idText = idText + saveTextAlign
            pycmd.pcmd(idText, id, None)
            inputS.append(id)
            eprint.p('\n')
        else:
            idTextIndex = int(text.getTextIndex(autoInfoText))
            fixIdWidth = textWidth - idTextIndex
            saveNoneAlign = text.align(saveNoneText, 'center', textWidth=fixIdWidth)
            idText = autoInfoText + saveNoneAlign
            eprint.p(idText)
            eprint.p('\n')
    else:
        pass
    return inputS

# 询问切换存档页面板
def askForChangeSavePagePanel(startId):
    cmdList = textload.getTextData(textload.cmdId,"changeSavePage")
    savePanelPage = str(cache.panelState['SeeSaveListPanel'])
    maxSavePanelPage = str(cache.maxSavePage)
    savePageText = '(' + savePanelPage + '/' + maxSavePanelPage + ')'
    eprint.printPageLine(sample='-',string=savePageText)
    eprint.p('\n')
    yrn = ans.optionint(None,3,askfor=False,cmdSize='center',startId=startId,cmdListData=cmdList)
    return yrn

# 询问覆盖存档面板
def askForOverlaySavePanel():
    cmdList = textload.getTextData(textload.cmdId,"overlaySave")
    messageText = textload.getTextData(textload.messageId,'21')
    eprint.pline()
    eprint.p(messageText)
    eprint.p('\n')
    yrn = ans.optionint(None,1,askfor=False,cmdListData=cmdList)
    return yrn