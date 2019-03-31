from script.Core import CacheContorl,EraPrint,TextLoading,PyCmd,GameInit,TextHandle
from script.Design import AttrCalculation,CmdButtonQueue

# 请求玩家输入姓名面板
def inputNamePanel():
    characterId = CacheContorl.characterData['characterId']
    CacheContorl.characterData['character'][characterId] = CacheContorl.temporaryCharacter.copy()
    AttrCalculation.setDefaultCache()
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messagePath, '4'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.currencymenu, 1)
    EraPrint.p('\n')
    return yrn

# 开始输入姓名
def startInputNamePanel():
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messagePath, '3'))
    inputState = 0
    while inputState == 0:
        characterName = GameInit.askfor_str()
        EraPrint.pl(characterName)
        if TextHandle.getTextIndex(characterName) > 10:
            EraPrint.pl(TextLoading.getTextData(TextLoading.errorPath, 'inputNameTooLongError'))
        else:
            inputState = 1
            CacheContorl.temporaryCharacter['Name'] = characterName

# 请求玩家输入昵称面板
def inputNickNamePanel():
    characterId = CacheContorl.characterData['characterId']
    CacheContorl.characterData['character'][characterId] = CacheContorl.temporaryCharacter.copy()
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messagePath, '6'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.inputnickname, 1)
    EraPrint.p('\n')
    return yrn

# 开始输入昵称面板
def startInputNickNamePanel():
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messagePath, '5'))
    inputState = 0
    while inputState == 0:
        characterNickName = GameInit.askfor_str()
        EraPrint.pl(characterNickName)
        if TextHandle.getTextIndex(characterNickName) > 10:
            EraPrint.pl(TextLoading.getTextData(TextLoading.errorPath, 'inputNickNameTooLongError'))
        else:
            inputState = 1
            CacheContorl.temporaryCharacter['NickName'] = characterNickName
    EraPrint.p('\n')

# 请求玩家输入自称面板
def inputSelfNamePanel():
    characterId = CacheContorl.characterData['characterId']
    PyCmd.clr_cmd()
    CacheContorl.characterData['character'][characterId] = CacheContorl.temporaryCharacter.copy()
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messagePath, '14'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.inputselfname, 1)
    EraPrint.p('\n')
    return yrn

# 开始输入自称
def startInputSelfName():
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messagePath, '15'))
    inputState = 0
    while inputState == 0:
        characterSelfName = GameInit.askfor_str()
        EraPrint.pl(characterSelfName)
        if TextHandle.getTextIndex(characterSelfName) > 10:
            EraPrint.pl(TextLoading.getTextData(TextLoading.errorPath, 'inputSelfNameTooLongError'))
        else:
            inputState = 1
            CacheContorl.temporaryCharacter['SelfName'] = characterSelfName
    EraPrint.p('\n')

# 请求玩家输入性别面板
def inputSexPanel():
    characterId = CacheContorl.characterData['characterId']
    sexId = CacheContorl.characterData['character'][characterId]['Sex']
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messagePath, '8')[sexId])
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.currencymenu, 1)
    EraPrint.p('\n')
    return yrn

# 玩家确认性别界面
def inputSexChoicePanel():
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messagePath, '7'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.sexmenu, 1)
    EraPrint.p('\n')
    return yrn

# 询问玩家是否进行详细设置面板
def attributeGenerationBranchPanel():
    characterId = CacheContorl.characterData['characterId']
    AttrCalculation.setAttrDefault(characterId)
    PyCmd.clr_cmd()
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messagePath, '9'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.currencymenu, 1)
    return yrn

# 详细设置属性1:询问玩家是否是小孩子
def detailedSetting1Panel():
    EraPrint.p('\n')
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messagePath, '10'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.detailedsetting1, 1)
    return yrn

# 详细设置属性2:询问玩家是否具备动物特征
def detailedSetting2Panel():
    EraPrint.p('\n')
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messagePath, '11'))
    yrn = CmdButtonQueue.optionstr(CmdButtonQueue.detailedsetting2, 5, 'center', True)
    return yrn

# 详细设置属性3:询问玩家是否具备丰富的性经验
def detailedSetting3Panel():
    EraPrint.p('\n')
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messagePath, '12'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.detailedsetting3)
    return yrn

# 详细设置属性4:询问玩家的胆量
def detailedSetting4Panel():
    EraPrint.p('\n')
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messagePath, '13'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.detailedsetting4)
    return yrn

# 详细设置属性5:询问玩家的性格
def detailedSetting5Panel():
    EraPrint.p('\n')
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messagePath, '16'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.detailedsetting5)
    return yrn

# 详细设置属性6:询问玩家的自信
def detailedSetting6Panel():
    EraPrint.p('\n')
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messagePath, '17'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.detailedsetting6)
    return yrn

# 详细设置属性7:询问玩家友善
def detailedSetting7Panel():
    EraPrint.p('\n')
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messagePath, '18'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.detailedsetting7)
    return yrn

# 详细设置属性8:询问玩家体型
def detailedSetting8Panel():
    EraPrint.p('\n')
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messagePath, '29'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.detailedsetting8)
    return yrn
