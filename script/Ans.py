import core.PyCmd as pycmd
import core.game as game
import script.TextLoading as textload
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

# 用于批量生成id命令
def optionint(cmdList,cmdColumn = 1,cmdSize = 'left',idSwitch = True):
    cmdListDate = textload.loadCmdAdv(cmdList).copy()
    inputI = []
    textWidth = config.text_width
    cmdIndex = int(textWidth/cmdColumn)
    for i in range(0,len(cmdListDate)):
        cmdText = dictionaries.handleText(cmdListDate[i])
        if idSwitch == True:
            id = idIndex(i)
        else:
            id = ''
        cmdTextAndId = id + cmdText
        cmdTextAndIdIndex = text.getTextIndex(cmdTextAndId)
        cmdfix = " " * (cmdIndex - cmdTextAndIdIndex)
        if cmdTextAndIdIndex < cmdIndex:
            if cmdSize == 'right':
                cmdTextAndId = cmdText + id
                if i == 0:
                    eprint.p(cmdfix)
                    pycmd.pcmd(cmdTextAndId, i, None)
                    inputI.append(str(i))
                elif i / cmdColumn >= 1 and i % cmdColumn == 0:
                    eprint.p('\n')
                    eprint.p(cmdfix)
                    cmdTextAndId = cmdTextAndId.rstrip()
                    pycmd.pcmd(cmdTextAndId, i, None)
                    inputI.append(str(i))
                else:
                    eprint.p(cmdfix)
                    pycmd.pcmd(cmdTextAndId, i, None)
                    inputI.append(str(i))
            elif cmdSize == 'left':
                cmdTextAndId = id + cmdText
                if i == 0:
                    pycmd.pcmd(cmdTextAndId, i, None)
                    eprint.p(cmdfix)
                    inputI.append(str(i))
                elif i / cmdColumn >= 1 and i % cmdColumn == 0:
                    eprint.p('\n')
                    cmdTextAndId = cmdTextAndId.rstrip()
                    pycmd.pcmd(cmdTextAndId, i, None)
                    inputI.append(str(i))
                else:
                    pycmd.pcmd(cmdTextAndId, i, None)
                    eprint.p(cmdfix)
                    inputI.append(str(i))
        else:
            pass
    eprint.p('\n')
    ans = int(game.askfor_All(inputI))
    return ans

# 用于批量生成文本命令
def optionstr(cmdList,cmdColumn = 1,cmdSize = 'left',lastLine = False):
    cmdListDate = textload.loadCmdAdv(cmdList)
    inputS = []
    textWidth = config.text_width
    cmdIndex = int(textWidth / cmdColumn)
    for i in range(0,len(cmdListDate)):
        cmdTextBak = dictionaries.handleText(cmdListDate[i])
        cmdText = '[' + cmdTextBak + ']'
        cmdTextWidth = text.getTextIndex(cmdText)
        cmdTextFix = ' ' * (cmdIndex - cmdTextWidth)
        if cmdSize == 'right':
            if i == 0:
                eprint.p(cmdTextFix)
                pycmd.pcmd(cmdText, cmdTextBak, None)
                inputS.append(cmdListDate[i])
            elif lastLine == False:
                if i / cmdColumn >= 1 and i % cmdColumn == 0:
                    eprint.p('\n')
                    eprint.p(cmdTextFix)
                    pycmd.pcmd(cmdText, cmdTextBak, None)
                    inputS.append(cmdTextBak)
                else:
                    eprint.p(cmdTextFix)
                    pycmd.pcmd(cmdText, cmdTextBak, None)
                    inputS.append(cmdTextBak)
            elif lastLine == True:
                if i / cmdColumn >= 1 and i % cmdColumn == 0:
                    eprint.p('\n')
                    eprint.p(cmdTextFix)
                    pycmd.pcmd(cmdText, cmdTextBak, None)
                    inputS.append(cmdTextBak)
                elif i == len(cmdListDate) - 1:
                    eprint.p('\n')
                    eprint.p(cmdTextFix)
                    pycmd.pcmd(cmdText, cmdTextBak, None)
                    inputS.append(cmdTextBak)
                else:
                    eprint.p(cmdTextFix)
                    pycmd.pcmd(cmdText, cmdTextBak, None)
                    inputS.append(cmdTextBak)
        elif cmdSize == 'left':
            if i == 0:
                pycmd.pcmd(cmdText, cmdTextBak, None)
                eprint.p(cmdTextFix)
                inputS.append(cmdTextBak)
            elif lastLine == False:
                if i / cmdColumn >= 1 and i % cmdColumn == 0:
                    eprint.p('\n')
                    pycmd.pcmd(cmdText, cmdTextBak, None)
                    inputS.append(cmdTextBak)
                else:
                    pycmd.pcmd(cmdText, cmdTextBak, None)
                    eprint.p(cmdTextFix)
                    inputS.append(cmdTextBak)
            elif lastLine == True:
                if i / cmdColumn >= 1 and i % cmdColumn == 0:
                    eprint.p('\n')
                    pycmd.pcmd(cmdText, cmdTextBak, None)
                    inputS.append(cmdTextBak)
                elif i == len(cmdListDate) - 1:
                    eprint.p('\n')
                    pycmd.pcmd(cmdText, cmdTextBak, None)
                    inputS.append(cmdTextBak)
                else:
                    pycmd.pcmd(cmdText, cmdTextBak, None)
                    eprint.p(cmdTextFix)
                    inputS.append(cmdTextBak)
        elif cmdSize == 'center':
            cmdTextFix = ' ' * int(cmdIndex/2 - cmdTextWidth/2)
            if i == 0:
                eprint.p(cmdTextFix)
                pycmd.pcmd(cmdText, cmdTextBak, None)
                eprint.p(cmdTextFix)
                inputS.append(cmdTextBak)
            elif lastLine == False:
                if i / cmdColumn >= 1 and i % cmdColumn == 0:
                    eprint.p('\n')
                    eprint.p(cmdTextFix)
                    pycmd.pcmd(cmdText, cmdTextBak, None)
                    inputS.append(cmdTextBak)
                else:
                    eprint.p(cmdTextFix)
                    pycmd.pcmd(cmdText, cmdTextBak, None)
                    eprint.p(cmdTextFix)
                    inputS.append(cmdTextBak)
            elif lastLine == True:
                if i / cmdColumn >= 1 and i % cmdColumn == 0:
                    eprint.p('\n')
                    eprint.p(cmdTextFix)
                    pycmd.pcmd(cmdText, cmdTextBak, None)
                    eprint.p(cmdTextFix)
                    inputS.append(cmdTextBak)
                elif i == len(cmdListDate) - 1:
                    eprint.p('\n')
                    eprint.p(cmdTextFix)
                    pycmd.pcmd(cmdText, cmdTextBak, None)
                    inputS.append(cmdTextBak)
                else:
                    eprint.p(cmdTextFix)
                    pycmd.pcmd(cmdText, cmdTextBak, None)
                    eprint.p(cmdTextFix)
                    inputS.append(cmdTextBak)
    eprint.p('\n')
    ans = game.askfor_All(inputS)
    return ans

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