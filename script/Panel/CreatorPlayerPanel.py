from script.Core import CacheContorl,EraPrint,TextLoading,PyCmd,GameInit,TextHandle
from script.Design import AttrCalculation,CmdButtonQueue

# 请求玩家输入姓名面板
def inputNamePanel():
    playerId = CacheContorl.playObject['objectId']
    CacheContorl.playObject['object'][playerId] = CacheContorl.temporaryObject.copy()
    AttrCalculation.setDefaultCache()
    EraPrint.p('\n')
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messageId, '4'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.currencymenu, 1)
    EraPrint.p('\n')
    return yrn

# 开始输入姓名
def startInputNamePanel():
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messageId, '3'))
    inputState = 0
    while inputState == 0:
        playerName = GameInit.askfor_str()
        EraPrint.pl(playerName)
        if TextHandle.getTextIndex(playerName) > 10:
            EraPrint.pl(TextLoading.getTextData(TextLoading.errorId, 'inputNameTooLongError'))
        else:
            inputState = 1
            CacheContorl.temporaryObject['Name'] = playerName

# 请求玩家输入昵称面板
def inputNickNamePanel():
    playerId = CacheContorl.playObject['objectId']
    CacheContorl.playObject['object'][playerId] = CacheContorl.temporaryObject.copy()
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messageId, '6'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.inputnickname, 1)
    EraPrint.p('\n')
    return yrn

# 开始输入昵称面板
def startInputNickNamePanel():
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messageId, '5'))
    inputState = 0
    while inputState == 0:
        playerNickName = GameInit.askfor_str()
        EraPrint.pl(playerNickName)
        if TextHandle.getTextIndex(playerNickName) > 10:
            EraPrint.pl(TextLoading.getTextData(TextLoading.errorId, 'inputNickNameTooLongError'))
        else:
            inputState = 1
            CacheContorl.temporaryObject['NickName'] = playerNickName
    EraPrint.p('\n')
    pass

# 请求玩家输入自称面板
def inputSelfNamePanel():
    playerId = CacheContorl.playObject['objectId']
    PyCmd.clr_cmd()
    CacheContorl.playObject['object'][playerId] = CacheContorl.temporaryObject.copy()
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messageId, '14'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.inputselfname, 1)
    EraPrint.p('\n')
    return yrn

# 开始输入自称
def startInputSelfName():
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messageId, '15'))
    inputState = 0
    while inputState == 0:
        playerSelfName = GameInit.askfor_str()
        EraPrint.pl(playerSelfName)
        if TextHandle.getTextIndex(playerSelfName) > 10:
            EraPrint.pl(TextLoading.getTextData(TextLoading.errorId, 'inputSelfNameTooLongError'))
        else:
            inputState = 1
            CacheContorl.temporaryObject['SelfName'] = playerSelfName
    EraPrint.p('\n')

# 请求玩家输入性别面板
def inputSexPanel():
    playerId = CacheContorl.playObject['objectId']
    sexId = CacheContorl.playObject['object'][playerId]['Sex']
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messageId, '8')[sexId])
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.currencymenu, 1)
    EraPrint.p('\n')
    return yrn

# 玩家确认性别界面
def inputSexChoicePanel():
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messageId, '7'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.sexmenu, 1)
    return yrn

# 询问玩家是否进行详细设置面板
def attributeGenerationBranchPanel():
    playerId = CacheContorl.playObject['objectId']
    AttrCalculation.setAttrDefault(playerId)
    PyCmd.clr_cmd()
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messageId, '9'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.currencymenu, 1)
    return yrn

# 详细设置属性1:询问玩家是否是小孩子
def detailedSetting1Panel():
    EraPrint.p('\n')
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messageId, '10'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.detailedsetting1, 1)
    return yrn

# 详细设置属性2:询问玩家是否具备动物特征
def detailedSetting2Panel():
    EraPrint.p('\n')
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messageId, '11'))
    yrn = CmdButtonQueue.optionstr(CmdButtonQueue.detailedsetting2, 5, 'center', True)
    return yrn

# 详细设置属性3:询问玩家是否具备丰富的性经验
def detailedSetting3Panel():
    EraPrint.p('\n')
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messageId, '12'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.detailedsetting3)
    return yrn

# 详细设置属性4:询问玩家的胆量
def detailedSetting4Panel():
    EraPrint.p('\n')
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messageId, '13'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.detailedsetting4)
    return yrn

# 详细设置属性5:询问玩家的性格
def detailedSetting5Panel():
    EraPrint.p('\n')
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messageId, '16'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.detailedsetting5)
    return yrn

# 详细设置属性6:询问玩家的自信
def detailedSetting6Panel():
    EraPrint.p('\n')
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messageId, '17'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.detailedsetting6)
    return yrn

# 详细设置属性7:询问玩家友善
def detailedSetting7Panel():
    EraPrint.p('\n')
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messageId, '18'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.detailedsetting7)
    return yrn

# 详细设置属性8:询问玩家体型
def detailedSetting8Panel():
    EraPrint.p('\n')
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messageId, '29'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.detailedsetting8)
    return yrn