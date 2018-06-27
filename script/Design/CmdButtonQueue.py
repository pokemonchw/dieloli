from script.Core import PyCmd,TextLoading,GameInit,EraPrint,TextHandle,GameConfig,Dictionaries

logomenu = "logoMenu"
currencymenu = "currencyMenu"
sexmenu = "sexMenu"
inputnickname = "inputNickName"
inputselfname = 'inputSelfName'
detailedsetting1 = "detailedSetting1"
detailedsetting2 = "detailedSetting2"
detailedsetting3 = "detailedSetting3"
detailedsetting4 = "detailedSetting4"
detailedsetting5 = "detailedSetting5"
detailedsetting6 = "detailedSetting6"
detailedsetting7 = "detailedSetting7"
detailedsetting8 = "detailedSetting8"
acknowledgmentAttribute = "acknowledgmentAttribute"
mainmenu = "mainMenu"
systemmenu = "systemMenu"
seeattrpanelmenu = "seeAttrPanelHandle"
changesavepage = "changeSavePage"
seeattronrverytime = "seeAttrOnEveryTime"
seeplayerlist = "seePlayerList"
inscenelist1 = "inSceneList1"
seemap = "seeMap"

# 用于批量生成id命令
def optionint(cmdList,cmdColumn = 1,idSize = 'left',idSwitch = True,askfor = True,cmdSize = 'left',startId = '0',cmdListData=None,lastLine = False):
    if cmdListData == None:
        cmdListData = TextLoading.getTextData(TextLoading.cmdId, cmdList).copy()
    else:
        pass
    inputI = []
    textWidth = GameConfig.text_width
    if lastLine == True:
        if len(cmdListData) < cmdColumn:
            cmdColumn = len(cmdListData)
    else:
        if len(cmdListData) + 1 < cmdColumn:
            cmdColumn = len(cmdListData)
    cmdIndex = int(textWidth/cmdColumn)
    if len(cmdListData) + 1 < cmdColumn:
        cmdColumn = len(cmdListData) + 1
    for i in range(0,len(cmdListData)):
        cmdText = Dictionaries.handleText(cmdListData[i])
        startId = int(startId)
        returnId = i + startId
        if idSwitch == True:
            id = idIndex(returnId)
        else:
            id = ''
        cmdTextAndId = id + cmdText
        cmdTextAndIdIndex = TextHandle.getTextIndex(cmdTextAndId)
        if cmdTextAndIdIndex < cmdIndex:
            if idSize == 'right':
                cmdTextAndId = cmdText + id
            elif idSize == 'left':
                cmdTextAndId = id + cmdText
            if i == 0:
                cmdTextAndId = cmdTextAndId.rstrip()
                cmdSizePrint(cmdTextAndId, returnId, None, cmdIndex, cmdSize)
                inputI.append(str(returnId))
            elif i / cmdColumn >= 1 and i % cmdColumn == 0:
                EraPrint.p('\n')
                cmdTextAndId = cmdTextAndId.rstrip()
                cmdSizePrint(cmdTextAndId, returnId, None, cmdIndex, cmdSize)
                inputI.append(str(returnId))
            elif i == len(cmdListData) and lastLine == True:
                EraPrint.p('\n')
                cmdTextAndId = cmdTextAndId.rstrip()
                cmdSizePrint(cmdTextAndId, returnId, None, cmdIndex, cmdSize)
                inputI.append(str(returnId))
            else:
                cmdTextAndId = cmdTextAndId.rstrip()
                cmdSizePrint(cmdTextAndId, returnId, None, cmdIndex, cmdSize)
                inputI.append(str(returnId))
        else:
            pass
    EraPrint.p('\n')
    if askfor == True:
        ans = int(GameInit.askfor_Int(inputI))
        return ans
    else:
        return inputI

# 用于批量生成文本命令
def optionstr(cmdList,cmdColumn = 1,cmdSize = 'left',lastLine = False,askfor = True,cmdListData=None):
    if cmdListData == None:
        cmdListData = TextLoading.getTextData(TextLoading.cmdId, cmdList).copy()
    else:
        pass
    inputS = []
    textWidth = GameConfig.text_width
    if lastLine == True:
        if len(cmdListData) - 1 < cmdColumn:
            cmdColumn = len(cmdListData) - 1
    else:
        if len(cmdListData) < cmdColumn:
            cmdColumn = len(cmdListData)
    cmdIndex = int(textWidth / cmdColumn)
    for i in range(0,len(cmdListData)):
        cmdTextBak = Dictionaries.handleText(cmdListData[i])
        cmdText = '[' + cmdTextBak + ']'
        if i == 0:
            cmdSizePrint(cmdText,cmdTextBak,None,cmdIndex,cmdSize)
            inputS.append(cmdListData[i])
        elif i / cmdColumn >= 1 and i % cmdColumn == 0:
            EraPrint.p('\n')
            cmdSizePrint(cmdText, cmdTextBak, None, cmdIndex, cmdSize)
            inputS.append(cmdTextBak)
        elif i == len(cmdListData) - 1 and lastLine == True:
            EraPrint.p('\n')
            cmdSizePrint(cmdText, cmdTextBak, None, cmdIndex, cmdSize)
            inputS.append(cmdTextBak)
        else:
            cmdSizePrint(cmdText, cmdTextBak, None, cmdIndex, cmdSize)
            inputS.append(cmdTextBak)
    EraPrint.p('\n')
    if askfor == True:
        ans = GameInit.askfor_All(inputS)
        return ans
    else:
        return inputS

# 生成id文本
def idIndex(id):
    if id -100 >= 0:
        idS = "[" + str(id) + "] "
        return idS
    elif id - 10 >= 0:
        if id == 0:
            idS = "[00" + str(id) + "] "
            return idS
        else:
            idS = "[0" + str(id) + "] "
            return idS
    else:
        idS = "[00" + str(id) + "] "
        return idS

# 命令对齐
def cmdSizePrint(cmdText,cmdTextBak,cmdEvent = None,textWidth = 0,cmdSize = 'left'):
    if cmdSize == 'left':
        cmdWidth = TextHandle.getTextIndex(cmdText)
        cmdTextFix = ' ' * (textWidth - cmdWidth)
        PyCmd.pcmd(cmdText, cmdTextBak, cmdEvent)
        EraPrint.p(cmdTextFix)
    elif cmdSize == 'center':
        cmdWidth = TextHandle.getTextIndex(cmdText)
        cmdTextFix = ' ' * (int(textWidth/2) - int(cmdWidth/2))
        EraPrint.p(cmdTextFix)
        PyCmd.pcmd(cmdText, cmdTextBak, cmdEvent)
        EraPrint.p(cmdTextFix)
    elif cmdSize == 'right':
        cmdWidth = TextHandle.getTextIndex(cmdText)
        cmdTextFix = ' ' * (textWidth - cmdWidth)
        EraPrint.p(cmdTextFix)
        PyCmd.pcmd(cmdText, cmdTextBak, cmdEvent)