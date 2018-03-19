import core.CacheContorl as cache
import core.GameConfig as config
import core.PyCmd as pycmd
import core.TextLoading as textload
import core.EraPrint as eprint
import script.Ans as ans
import core.TextHandle as text

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
    if savePanelPage == int(config.save_page) + 1:
        startSaveId = int(pageSaveValue) * savePanelPage + 1
        overSaveId = startSaveId + lastSavePageValue
    else:
        overSaveId = int(pageSaveValue) * savePanelPage
        startSaveId = overSaveId - int(pageSaveValue)
    for i in range(startSaveId,overSaveId):
        id = ans.idIndex(i)
        idText = id + idInfoText  + " " + str(i) + ":"
        idTextList.append(idText)
    for i in range(startSaveId,overSaveId):
        id = str(i)
        idText = idTextList[i]
        idTextIndex = int(text.getTextIndex(idText))
        fixIdText = ' ' * (textWidth - idTextIndex)
        idText = idText + fixIdText
        eprint.plittleline()
        pycmd.pcmd(idText,id,None)
        inputS.append(id)
        eprint.p('\n')
    if autoSave == True:
        autoInfoText = textload.getTextData(textload.stageWordId,"73")
        i = pageSaveValue
        id = ans.idIndex(i)
        idText = id + autoInfoText
        idTextIndex = int(text.getTextIndex(idText))
        fixIdText = ' ' * (textWidth - idTextIndex)
        idText = idText + fixIdText
        eprint.plittleline()
        pycmd.pcmd(idText, id, None)
        inputS.append(id)
        eprint.p('\n')
    else:
        pass
    eprint.pline()
    return inputS

def askForChangeSavePagePanel(startId):
    inputS = []
    cmdList = textload.getTextData(textload.cmdId,"changeSavePage")
    savePanelPage = str(cache.panelState['SeeSaveListPanel'])
    maxSavePanelPage = str(cache.maxSavePage)
    upPage = cmdList[0]
    downPage = cmdList[1]
    backButton = cmdList[2]
    upPageId = ans.idIndex(startId)
    upPageText = upPageId + upPage
    upPageFix = text.align(upPageText,just='center',onlyFix=True,columns=3)
    eprint.p(upPageFix)
    pycmd.pcmd(upPageText,startId,None)
    inputS.append(startId)
    eprint.p(upPageFix)
    savePageText = '(' + savePanelPage + '/' + maxSavePanelPage + ')'
    savePageFix = text.align(savePageText,just='center',onlyFix=True,columns=3)
    eprint.p(savePageFix)
    eprint.p(savePageText)
    eprint.p(savePageFix)
    downPageId = ans.idIndex(startId + 1)
    downPageText = downPageId + downPage
    downPageFix = text.align(downPageText, just='center', onlyFix=True, columns=3)
    eprint.p(downPageFix)
    pycmd.pcmd(downPageText,startId + 1,None)
    eprint.p(downPageFix)
    eprint.p('\n')
    inputS.append(startId + 1)
    backButtonId = ans.idIndex(startId + 2)
    backButtonText = backButtonId + backButton
    backButtonFix = text.align(backButtonText,just='center',onlyFix=True)
    eprint.p(backButtonFix)
    pycmd.pcmd(backButtonText,startId + 2,None)
    eprint.p(backButtonFix)
    inputS.append(startId + 2)
    return inputS