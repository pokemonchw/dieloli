import core.PyCmd as pycmd
import core.game as game
import core.TextLoading as textload
import core.EraPrint as eprint
import core.TextHandle as text
import core.GameConfig as config
import core.Dictionaries as dictionaries

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
detailedsetting7 = "detailedSettind7"
acknowledgmentAttribute = "acknowledgmentAttribute"
mainmenu = "mainMenu"
systemmenu = "systemMenu"
seeattrpanelmenu = "seeAttrPanelHandle"
changesavepage = "changeSavePage"
seeattronrverytime = "seeAttrOnEveryTime"
seeplayerlist = "seePlayerList"

# 用于批量生成id命令
def optionint(cmdList,cmdColumn = 1,idSize = 'left',idSwitch = True,askfor = True,cmdSize = 'left',startId = '0',cmdListData=None):
    if cmdListData == None:
        cmdListData = textload.getTextData(textload.cmdId, cmdList).copy()
    else:
        pass
    inputI = []
    textWidth = config.text_width
    cmdIndex = int(textWidth/cmdColumn)
    if len(cmdListData) < cmdColumn:
        cmdColumn = len(cmdListData) - 1
    for i in range(0,len(cmdListData)):
        cmdText = dictionaries.handleText(cmdListData[i])
        startId = int(startId)
        returnId = i + startId
        if idSwitch == True:
            id = idIndex(returnId)
        else:
            id = ''
        cmdTextAndId = id + cmdText
        cmdTextAndIdIndex = text.getTextIndex(cmdTextAndId)
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
                eprint.p('\n')
                cmdTextAndId = cmdTextAndId.rstrip()
                cmdSizePrint(cmdTextAndId, returnId, None, cmdIndex, cmdSize)
                inputI.append(str(returnId))
            else:
                cmdTextAndId = cmdTextAndId.rstrip()
                cmdSizePrint(cmdTextAndId, returnId, None, cmdIndex, cmdSize)
                inputI.append(str(returnId))
        else:
            pass
    eprint.p('\n')
    if askfor == True:
        ans = int(game.askfor_Int(inputI))
        return ans
    else:
        return inputI

# 用于批量生成文本命令
def optionstr(cmdList,cmdColumn = 1,cmdSize = 'left',lastLine = False,askfor = True,cmdListData=None):
    if cmdListData == None:
        cmdListData = textload.getTextData(textload.cmdId, cmdList).copy()
    else:
        pass
    inputS = []
    textWidth = config.text_width
    if lastLine == True:
        if len(cmdListData) - 1 < cmdColumn:
            cmdColumn = len(cmdListData) - 1
    else:
        if len(cmdListData) < cmdColumn:
            cmdColumn = len(cmdListData) - 1
    cmdIndex = int(textWidth / cmdColumn)
    for i in range(0,len(cmdListData)):
        cmdTextBak = dictionaries.handleText(cmdListData[i])
        cmdText = '[' + cmdTextBak + ']'
        if i == 0:
            cmdSizePrint(cmdText,cmdTextBak,None,cmdIndex,cmdSize)
            inputS.append(cmdListData[i])
        elif i / cmdColumn >= 1 and i % cmdColumn == 0:
            eprint.p('\n')
            cmdSizePrint(cmdText, cmdTextBak, None, cmdIndex, cmdSize)
            inputS.append(cmdTextBak)
        elif i == len(cmdListData) - 1 and lastLine == True:
            eprint.p('\n')
            cmdSizePrint(cmdText, cmdTextBak, None, cmdIndex, cmdSize)
            inputS.append(cmdTextBak)
        else:
            cmdSizePrint(cmdText, cmdTextBak, None, cmdIndex, cmdSize)
            inputS.append(cmdTextBak)
    eprint.p('\n')
    if askfor == True:
        ans = game.askfor_All(inputS)
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
        cmdWidth = text.getTextIndex(cmdText)
        cmdTextFix = ' ' * (textWidth - cmdWidth)
        pycmd.pcmd(cmdText, cmdTextBak, cmdEvent)
        eprint.p(cmdTextFix)
    elif cmdSize == 'center':
        cmdWidth = text.getTextIndex(cmdText)
        cmdTextFix = ' ' * (int(textWidth/2) - int(cmdWidth/2))
        eprint.p(cmdTextFix)
        pycmd.pcmd(cmdText, cmdTextBak, cmdEvent)
        eprint.p(cmdTextFix)
    elif cmdSize == 'right':
        cmdWidth = text.getTextIndex(cmdText)
        cmdTextFix = ' ' * (textWidth - cmdWidth)
        eprint.p(cmdTextFix)
        pycmd.pcmd(cmdText, cmdTextBak, cmdEvent)