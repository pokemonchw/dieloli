import core.PyCmd as pycmd
import core.game as game
import script.TextLoading as text
import core.EraPrint as eprint

logomenu = "logoMenu"
currencymenu = "currencyMenu"
sexmenu = "sexMenu"
inputnickname = "inputNickName"
detailedsetting1 = "detailedSetting1"
detailedsetting2 = "detailedSetting2"

def optionint(cmdList,cmdColumn,idSwitch = True):
    cmdListDate = text.loadCmdAdv(cmdList)
    inputI = []
    for i in range(0,len(cmdListDate)):
        if idSwitch == True:
            id = idIndex(i)
        else:
            id = ''
        if i == 0:
            pycmd.pcmd(id + cmdListDate[i], i, None)
            inputI.append(str(i))
        else:
            if i / cmdColumn >= 1 and i % cmdColumn == 0:
                eprint.p('\n')
                pycmd.pcmd(id + cmdListDate[i], i, None)
                inputI.append(str(i))
            else:
                pycmd.pcmd(id + cmdListDate[i], i, None)
                inputI.append(str(i))
    eprint.p('\n')
    ans = int(game.askfor_All(inputI))
    return ans

def optionstr(cmdList,cmdColumn):
    cmdListDate = text.loadCmdAdv(cmdList)
    inputS = []
    for i in range(0,len(cmdListDate)):
        if i == 0:
            pycmd.pcmd(cmdListDate[i], cmdListDate[i], None)
            inputS.append(cmdListDate[i])
        else:
            if i / cmdColumn >= 1 and i % cmdColumn == 0:
                eprint.p('\n')
                pycmd.pcmd(cmdListDate[i], cmdListDate[i], None)
                inputS.append(cmdListDate[i])
            else:
                pycmd.pcmd(cmdListDate[i], cmdListDate[i], None)
                inputS.append(cmdListDate[i])
    eprint.p('\n')
    ans = game.askfor_All(inputS)
    return ans

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