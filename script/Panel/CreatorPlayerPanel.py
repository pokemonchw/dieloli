import core.CacheContorl as cache
import design.AttrCalculation as attr
import core.EraPrint as eprint
import core.TextLoading as textload
import core.PyCmd as pycmd
import core.game as game
import core.TextHandle as text
import design.Ans as ans

# 请求玩家输入姓名面板
def inputNamePanel():
    playerId = cache.playObject['objectId']
    cache.playObject['object'][playerId] = cache.temporaryObject.copy()
    attr.setDefaultCache()
    eprint.p('\n')
    eprint.pline()
    eprint.pl(textload.getTextData(textload.messageId, '4'))
    yrn = ans.optionint(ans.currencymenu, 1)
    eprint.p('\n')
    return yrn

# 开始输入姓名
def startInputNamePanel():
    eprint.pline()
    eprint.pl(textload.getTextData(textload.messageId, '3'))
    inputState = 0
    while inputState == 0:
        playerName = game.askfor_str()
        eprint.pl(playerName)
        if text.getTextIndex(playerName) > 10:
            eprint.pl(textload.getTextData(textload.errorId, 'inputNameTooLongError'))
        else:
            inputState = 1
            cache.temporaryObject['Name'] = playerName

# 请求玩家输入昵称面板
def inputNickNamePanel():
    playerId = cache.playObject['objectId']
    cache.playObject['object'][playerId] = cache.temporaryObject.copy()
    eprint.pline()
    eprint.pl(textload.getTextData(textload.messageId, '6'))
    yrn = Ans.optionint(Ans.inputnickname, 1)
    eprint.p('\n')
    return yrn

# 开始输入昵称面板
def startInputNickNamePanel():
    eprint.pline()
    eprint.pl(textload.getTextData(textload.messageId, '5'))
    inputState = 0
    while inputState == 0:
        playerNickName = game.askfor_str()
        eprint.pl(playerNickName)
        if text.getTextIndex(playerNickName) > 10:
            eprint.pl(textload.getTextData(textload.errorId, 'inputNickNameTooLongError'))
        else:
            inputState = 1
            cache.temporaryObject['NickName'] = playerNickName
    eprint.p('\n')
    pass

# 请求玩家输入自称面板
def inputSelfNamePanel():
    playerId = cache.playObject['objectId']
    pycmd.clr_cmd()
    cache.playObject['object'][playerId] = cache.temporaryObject.copy()
    eprint.pline()
    eprint.pl(textload.getTextData(textload.messageId, '14'))
    yrn = Ans.optionint(Ans.inputselfname, 1)
    eprint.p('\n')
    return yrn

# 开始输入自称
def startInputSelfName():
    eprint.pline()
    eprint.pl(textload.getTextData(textload.messageId, '15'))
    inputState = 0
    while inputState == 0:
        playerSelfName = game.askfor_str()
        eprint.pl(playerSelfName)
        if text.getTextIndex(playerSelfName) > 10:
            eprint.pl(textload.getTextData(textload.errorId, 'inputSelfNameTooLongError'))
        else:
            inputState = 1
            cache.temporaryObject['SelfName'] = playerSelfName
    eprint.p('\n')

# 请求玩家输入性别面板
def inputSexPanel():
    playerId = cache.playObject['objectId']
    sexId = cache.playObject['object'][playerId]['Sex']
    eprint.pline()
    eprint.pl(textload.getTextData(textload.messageId, '8')[sexId])
    yrn = Ans.optionint(Ans.currencymenu, 1)
    eprint.p('\n')
    return yrn

# 玩家确认性别界面
def inputSexChoicePanel():
    eprint.pline()
    eprint.pl(textload.getTextData(textload.messageId, '7'))
    yrn = Ans.optionint(Ans.sexmenu, 1)
    return yrn

# 询问玩家是否进行详细设置面板
def attributeGenerationBranchPanel():
    playerId = cache.playObject['objectId']
    attr.setAttrDefault(playerId)
    pycmd.clr_cmd()
    eprint.pline()
    eprint.pl(textload.getTextData(textload.messageId, '9'))
    yrn = Ans.optionint(Ans.currencymenu, 1)
    return yrn

# 详细设置属性1:询问玩家是否是小孩子
def detailedSetting1Panel():
    eprint.p('\n')
    eprint.pline()
    eprint.pl(textload.getTextData(textload.messageId, '10'))
    yrn = Ans.optionint(Ans.detailedsetting1, 1)
    return yrn

# 详细设置属性2:询问玩家是否具备动物特征
def detailedSetting2Panel():
    eprint.p('\n')
    eprint.pline()
    eprint.pl(textload.getTextData(textload.messageId, '11'))
    yrn = Ans.optionstr(Ans.detailedsetting2, 5, 'center', True)
    return yrn

# 详细设置属性3:询问玩家是否具备丰富的性经验
def detailedSetting3Panel():
    eprint.p('\n')
    eprint.pline()
    eprint.pl(textload.getTextData(textload.messageId, '12'))
    yrn = Ans.optionint(Ans.detailedsetting3)
    return yrn

# 详细设置属性4:询问玩家的胆量
def detailedSetting4Panel():
    eprint.p('\n')
    eprint.pline()
    eprint.pl(textload.getTextData(textload.messageId, '13'))
    yrn = Ans.optionint(Ans.detailedsetting4)
    return yrn

# 详细设置属性5:询问玩家的性格
def detailedSetting5Panel():
    eprint.p('\n')
    eprint.pline()
    eprint.pl(textload.getTextData(textload.messageId, '16'))
    yrn = Ans.optionint(Ans.detailedsetting5)
    return yrn

# 详细设置属性6:询问玩家的自信
def detailedSetting6Panel():
    eprint.p('\n')
    eprint.pline()
    eprint.pl(textload.getTextData(textload.messageId, '17'))
    yrn = Ans.optionint(Ans.detailedsetting6)
    return yrn

# 详细设置属性7:询问玩家友善
def detailedSetting7Panel():
    eprint.p('\n')
    eprint.pline()
    eprint.pl(textload.getTextData(textload.messageId, '18'))
    yrn = Ans.optionint(Ans.detailedsetting7)
    return yrn