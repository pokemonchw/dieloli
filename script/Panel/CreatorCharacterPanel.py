from script.Core import CacheContorl,EraPrint,TextLoading,PyCmd,GameInit,TextHandle
from script.Design import AttrCalculation,CmdButtonQueue

def inputNamePanel() -> str:
    '''
    请求玩家输入姓名面板
    '''
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messagePath, '4'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.currencymenu)
    EraPrint.p('\n')
    return yrn

def startInputNamePanel():
    '''
    玩家姓名输入处理面板
    '''
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

def inputNickNamePanel() -> str:
    '''
    请求玩家输入昵称面板
    '''
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messagePath, '6'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.inputnickname)
    EraPrint.p('\n')
    return yrn

def startInputNickNamePanel():
    '''
    玩家昵称输入处理面板
    '''
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

def inputSelfNamePanel() -> str:
    '''
    请求玩家输入自称面板
    '''
    PyCmd.clr_cmd()
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messagePath, '14'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.inputselfname)
    EraPrint.p('\n')
    return yrn

def startInputSelfName():
    '''
    玩家自称输入处理面板
    '''
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

def inputSexPanel() -> str:
    '''
    请求玩家选择性别面板
    '''
    characterId = CacheContorl.characterData['characterId']
    sexId = CacheContorl.characterData['character'][characterId].Sex
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messagePath, '8')[sexId])
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.currencymenu)
    EraPrint.p('\n')
    return yrn

def inputSexChoicePanel() -> str:
    '''
    玩家性别选择面板
    '''
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messagePath, '7'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.sexmenu)
    EraPrint.p('\n')
    return yrn

def attributeGenerationBranchPanel() -> str:
    '''
    玩家确认进行详细设置面板
    '''
    PyCmd.clr_cmd()
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messagePath, '9'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.currencymenu)
    return yrn

def detailedSetting1Panel() -> str:
    '''
    询问玩家年龄模板面板
    '''
    EraPrint.p('\n')
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messagePath, '10'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.detailedsetting1)
    return yrn

def detailedSetting3Panel() -> str:
    '''
    询问玩家性经验程度面板
    '''
    EraPrint.p('\n')
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messagePath, '12'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.detailedsetting3)
    return yrn

def detailedSetting8Panel() -> str:
    '''
    询问玩家肥胖程度面板
    '''
    EraPrint.p('\n')
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messagePath, '29'))
    yrn = CmdButtonQueue.optionint(CmdButtonQueue.detailedsetting8)
    return yrn

def enterCharacterNatureHead():
    '''
    用于确认角色性格的头部面板
    '''
    EraPrint.p('\n')
    EraPrint.pline()
    EraPrint.pl(TextLoading.getTextData(TextLoading.messagePath, '39'))

def enterCharacterNatureEnd() -> list:
    '''
    用户确认角色性格的尾部面板
    Return arguments:
    list -- 按钮列表
    '''
    EraPrint.pline()
    return CmdButtonQueue.optionint(CmdButtonQueue.entercharacternaturepanel,1,'left',True,False)
