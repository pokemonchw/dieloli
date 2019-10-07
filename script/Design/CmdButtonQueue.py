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
detailedsetting9 = "detailedsetting9"
acknowledgmentAttribute = "acknowledgmentAttribute"
mainmenu = "mainMenu"
systemmenu = "systemMenu"
seeattrpanelmenu = "seeAttrPanelHandle"
changesavepage = "changeSavePage"
seeattronrverytime = "seeAttrOnEveryTime"
seecharacterlist = "seeCharacterList"
inscenelist1 = "inSceneList1"
seemap = "seeMap"
gamehelp = "gameHelp"
seecharacterwearclothes = "seeCharacterWearClothes"
changescenecharacterlist = 'changeSceneCharacterList'
seecharacterclothes = 'seeCharacterClothes'
askseeclothinginfopanel = 'askSeeClothingInfoPanel'
seeclothinginfoaskpanel = 'seeClothingInfoAskPanel'

def optionint(
    cmdList:list,
        cmdColumn = 1,
        idSize = 'left',
        idSwitch = True,
        askfor = True,
        cmdSize = 'left',
        startId = 0,
        cmdListData=None,
        lastLine = False) -> list:
    '''
    批量绘制带id命令列表
    例:
    [000]开始游戏
    Keyword arguments:
    cmdList -- 命令列表id，当cmdListData为None时，根据此id调用cmdList内的命令数据
    cmdColumn -- 每行命令列数 (default 1)
    idSize -- id文本位置(left/center/right) (default 'left')
    idSwitch -- id显示开关 (default True)
    askfor -- 绘制完成时等待玩家输入的开关 (default True)
    cmdSize -- 命令文本在当前列的对齐方式(left/center/right) (default 'left')
    startId -- 命令列表的起始id (default 0)
    cmdListData -- 命令列表数据 (default None)
    lastLine -- 最后一个命令换行绘制 (default False)
    '''
    if cmdListData == None:
        cmdListData = TextLoading.getTextData(TextLoading.cmdPath, cmdList).copy()
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
    EraPrint.p('\n')
    if askfor == True:
        ans = int(GameInit.askfor_Int(inputI))
        return ans
    else:
        return inputI

def optionstr(
    cmdList:str,
        cmdColumn = 1,
        cmdSize = 'left',
        lastLine = False,
        askfor = True,
        cmdListData = None,
        nullCmd = '',
        returnData = None) -> list:
    '''
    绘制无id的文本命令列表
    例:
    [长寿的青蛙]
    Keyword arguments:
    cmdList -- 命令列表id，当cmdListData为None时，根据此id调用cmdList内的命令数据
    cmdColumn -- 每行命令列数 (default 1)
    cmdSize -- 命令文本在当前列的对齐方式(left/center/right) (default 'left')
    lastLine -- 最后一个命令换行绘制 (default False)
    cmdListData -- 命令列表数据 (default None)
    nullCmd -- 在列表中按纯文本绘制，并不加入监听列表的命令文本
    returnData -- 命令返回数据 (default None)
    '''
    if cmdListData == None:
        cmdListData = TextLoading.getTextData(TextLoading.cmdPath, cmdList).copy()
    inputS = []
    textWidth = GameConfig.text_width
    if lastLine == True:
        if len(cmdListData) - 1 < cmdColumn:
            cmdColumn = len(cmdListData) - 1
    else:
        if len(cmdListData) < cmdColumn:
            cmdColumn = len(cmdListData)
    cmdIndex = int(textWidth / cmdColumn)
    nowNullCmd = nullCmd
    for i in range(0,len(cmdListData)):
        nowNullCmd = True
        if returnData == None:
            if nullCmd == cmdListData[i]:
                nowNullCmd = False
            cmdTextBak = Dictionaries.handleText(cmdListData[i])
            cmdText = '[' + cmdTextBak + ']'
        else:
            if nullCmd == returnData[i]:
                nowNullCmd = False
            cmdTextBak = returnData[i]
            cmdText = '[' + cmdListData[i] + ']'
        if i == 0:
            cmdSizePrint(cmdText,cmdTextBak,None,cmdIndex,cmdSize,noNullCmd=nowNullCmd)
            if nowNullCmd:
                inputS.append(cmdListData[i])
        elif i / cmdColumn >= 1 and i % cmdColumn == 0:
            EraPrint.p('\n')
            cmdSizePrint(cmdText,cmdTextBak,None,cmdIndex,cmdSize,noNullCmd=nowNullCmd)
            if nowNullCmd:
                inputS.append(cmdTextBak)
        elif i == len(cmdListData) - 1 and lastLine == True:
            EraPrint.p('\n')
            cmdSizePrint(cmdText,cmdTextBak,None,cmdIndex,cmdSize,noNullCmd=nowNullCmd)
            if nowNullCmd:
                inputS.append(cmdTextBak)
        else:
            cmdSizePrint(cmdText,cmdTextBak,None,cmdIndex,cmdSize,noNullCmd=nowNullCmd)
            if nowNullCmd:
                inputS.append(cmdTextBak)
    EraPrint.p('\n')
    if askfor == True:
        ans = GameInit.askfor_All(inputS)
        return ans
    else:
        return inputS

def idIndex(id):
    '''
    生成命令id文本
    Keyword arguments:
    id -- 命令id
    '''
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

def cmdSizePrint(cmdText,cmdTextBak,cmdEvent = None,textWidth = 0,cmdSize = 'left',noNullCmd=True):
    '''
    计算命令对齐方式，补全文本并绘制
    Keyword arguments:
    cmdText -- 命令文本
    cmdTextBak -- 命令被触发时返回的文本
    cmdEvent -- 命令绑定的事件 (default None)
    textWidth -- 文本对齐时补全空间宽度
    cmdSize -- 命令对齐方式(left/center/right) (default 'left')
    noNullCmd -- 绘制命令而非null命令样式的文本 (default False)
    '''
    if noNullCmd == False:
        cmdText = '<nullcmd>' + cmdText + '</nullcmd>'
    if cmdSize == 'left':
        cmdWidth = TextHandle.getTextIndex(cmdText)
        cmdTextFix = ' ' * (textWidth - cmdWidth)
        if noNullCmd:
            PyCmd.pcmd(cmdText, cmdTextBak, cmdEvent)
        else:
            EraPrint.p(cmdText)
        EraPrint.p(cmdTextFix)
    elif cmdSize == 'center':
        cmdWidth = TextHandle.getTextIndex(cmdText)
        cmdTextFix = ' ' * (int(textWidth/2) - int(cmdWidth/2))
        EraPrint.p(cmdTextFix)
        if noNullCmd:
            PyCmd.pcmd(cmdText, cmdTextBak, cmdEvent)
        else:
            EraPrint.p(cmdText)
        EraPrint.p(cmdTextFix)
    elif cmdSize == 'right':
        cmdWidth = TextHandle.getTextIndex(cmdText)
        cmdTextFix = ' ' * (textWidth - cmdWidth)
        if noNullCmd:
            PyCmd.pcmd(cmdText, cmdTextBak, cmdEvent)
        else:
            EraPrint.p(cmdText)
        EraPrint.p(cmdTextFix)
